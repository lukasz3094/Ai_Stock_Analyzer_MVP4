# train_gnn.py
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import dense_to_sparse

from app.branches.prediction_branch.data_loader.features_loader import get_data_for_model_per_company
from app.services.getters.companies_getter import get_companies_by_group

# ======================
# Konfiguracja ogólna
# ======================
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

Y_CONFIG = {
    "horizon": 5,
    "style": "log",              # log-return
    "winsorize": 0.005,          # przytnij ogony
    "neutralize_sector": False,  # przy GNN zwykle nie neutralizujemy tu
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# Helpery
# ======================
def _winsorize(s: pd.Series, alpha: float) -> pd.Series:
    lo = s.quantile(alpha)
    hi = s.quantile(1 - alpha)
    return s.clip(lower=lo, upper=hi)

def compute_forward_return(df: pd.DataFrame, horizon: int, style: str = "log") -> pd.Series:
    df = df.sort_values(["company_id", "date"]).copy()
    s = pd.to_numeric(df["close"], errors="coerce")
    fwd = df.groupby("company_id")["close"].shift(-horizon) / s
    if style == "log":
        y = np.log(fwd)
    else:
        y = fwd - 1.0
    return y

def safe_log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    clean = {k: float(v) for k, v in metrics.items() if np.isfinite(v)}
    if clean:
        mlflow.log_metrics(clean, step=step)

def numericize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ======================
# Loader panelu danych
# ======================
def load_panel(company_ids: List[int]) -> pd.DataFrame:
    frames = []
    for cid in company_ids:
        df = get_data_for_model_per_company(cid)
        if df is None or df.empty:
            continue

        df = df.loc[:, ~df.isna().all()]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)

    all_na_cols = [c for c in out.columns if out[c].isna().all()]
    if all_na_cols:
        out = out.drop(columns=all_na_cols)

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out

# ======================
# Wybór cech
# ======================
BLACKLIST = {
    "id", "company_id", "date", "sector_id", "created_at",
    "y", "date_target", "close_true_target", "close_pred_target"
}
REQUIRED = {"close"}

def infer_common_feature_cols(df: pd.DataFrame, min_presence_ratio: float = 0.8) -> List[str]:
    missing_required = [c for c in REQUIRED if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    candidates = [c for c in numeric_cols if c not in BLACKLIST]

    last = df.sort_values(["company_id", "date"]).groupby("company_id").tail(1)
    good = []
    n_companies = last["company_id"].nunique()
    for c in candidates:
        present = last[c].notna().sum()
        if n_companies == 0:
            continue
        if present / n_companies >= min_presence_ratio:
            good.append(c)

    if not good:
        # fallback: min 50%
        for c in candidates:
            present = last[c].notna().sum()
            if n_companies and present / n_companies >= 0.5:
                good.append(c)

    if not good:
        raise ValueError("No common feature columns inferred — data too sparse.")

    return sorted(good)

# ======================
# Budowa grafu
# ======================
def build_rolling_corr_adj(df_train: pd.DataFrame, window: int = 120, thr: float = 0.5) -> Tuple[np.ndarray, List[int]]:
    """
    Macierz sąsiedztwa na podstawie korelacji log-zwrotów (rolling lub statycznej fallback).
    Zwraca adj (NxN) i listę company_id odpowiadającą indeksom w adj.
    """
    if df_train.empty:
        return np.zeros((0, 0), dtype=np.int64), []

    tmp = df_train.sort_values(["company_id", "date"]).copy()

    tmp["close"] = pd.to_numeric(tmp["close"], errors="coerce")
    tmp["log_ret"] = np.log(tmp["close"].clip(lower=1e-12)).groupby(tmp["company_id"]).diff()

    pivot = tmp.pivot(index="date", columns="company_id", values="log_ret")

    # jeśli danych mało do rolling, użyj zwykłej korelacji jako fallback
    if pivot.shape[0] < window:
        corr = pivot.corr(min_periods=20)
    else:
        rc = pivot.rolling(window).corr()
        # ostatnia data z pełnym oknem
        last_date = rc.index.get_level_values(0).max()
        corr = rc.xs(last_date, level=0)

    corr = corr.fillna(0.0)

    company_ids = list(corr.columns.astype(int))
    A = (corr.to_numpy() >= thr).astype(np.int64)
    np.fill_diagonal(A, 0)
    return A, company_ids

def add_sector_edges(adj: np.ndarray, company_ids: List[int], df_any: pd.DataFrame) -> np.ndarray:
    """Dodaj krawędzie sektorowe: jeśli spółki w tym samym sektorze → 1."""
    meta = (df_any.groupby("company_id")["sector_id"]
                  .agg(lambda s: s.dropna().iloc[-1] if len(s.dropna()) else np.nan)
                  .reindex(company_ids))
    sector = meta.to_numpy()
    same_sector = (sector[:, None] == sector[None, :]).astype(int)
    adj2 = np.maximum(adj, same_sector)
    np.fill_diagonal(adj2, 0)
    return adj2

# ======================
# Dataset / Tensorizacja
# ======================
def make_snapshot_tensor(
    df: pd.DataFrame,
    feature_cols: List[str],
    date_filter_mask: pd.Series
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    # weź tylko rekordy zgodne z maską (tu już y jest finite, jeśli maska tak zdefiniowana)
    snap = df.loc[date_filter_mask].sort_values(["company_id", "date"]).copy()
    if snap.empty:
        return torch.zeros((0, len(feature_cols)), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32), []

    # ostatni wiersz z ważnym y per company
    # (gdyby ktoś zawołał bez valid_y w masce, to i tak przefiltrujemy)
    snap["y"] = pd.to_numeric(snap["y"], errors="coerce")
    snap = snap[np.isfinite(snap["y"].to_numpy())]

    if snap.empty:
        return torch.zeros((0, len(feature_cols)), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32), []

    last = snap.groupby("company_id").tail(1).copy()

    # konwersja i filtracja niefinicznych CECH
    for c in feature_cols + ["y"]:
        last[c] = pd.to_numeric(last[c], errors="coerce")
    mask_finite = np.isfinite(last[feature_cols].to_numpy()).all(axis=1) & np.isfinite(last["y"].to_numpy())
    last = last.loc[mask_finite].copy()
    if last.empty:
        return torch.zeros((0, len(feature_cols)), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32), []

    X = last[feature_cols].astype(float).to_numpy()
    y = last["y"].astype(float).to_numpy()
    cids = last["company_id"].astype(int).tolist()

    return torch.from_numpy(X).float(), torch.from_numpy(y).float(), cids

# ======================
# Model: wspólny encoder + per-company heady
# ======================
class HeadedGNN(nn.Module):
    """
    Wspólny encoder GNN + osobny liniowy head dla każdej spółki.
    """
    def __init__(self, in_dim: int, hidden: int, company_ids: List[int], kind: str = "gcn"):
        super().__init__()
        if kind.lower() == "gat":
            self.c1 = GATConv(in_dim, hidden, heads=4, concat=False)
            self.c2 = GATConv(hidden, hidden, heads=4, concat=False)
        else:
            self.c1 = GCNConv(in_dim, hidden)
            self.c2 = GCNConv(hidden, hidden)
        self.act = nn.ReLU()

        self.heads = nn.ModuleDict({
            str(cid): nn.Linear(hidden, 1) for cid in company_ids
        })
        self.company_ids = company_ids

    def encode(self, x, edge_index):
        h = self.act(self.c1(x, edge_index))
        h = self.act(self.c2(h, edge_index))
        return h  # [N, hidden]

    def forward(self, x, edge_index):
        return self.encode(x, edge_index)

    def predict_for_indices(self, h, idx_tensor, cids_order: List[int]) -> torch.Tensor:
        preds = []
        for i in idx_tensor.tolist():
            cid = cids_order[i]
            head = self.heads[str(cid)]
            preds.append(head(h[i]).squeeze(-1))
        return torch.stack(preds, dim=0)

# ======================
# Główna pętla treningowa
# ======================
def train_one_snapshot(
    df: pd.DataFrame,
    date_split: str,
    corr_window: int = 120,
    corr_thr: float = 0.5,
    lr: float = 1e-3,
    epochs: int = 50,
    model_kind: str = "gcn",
    tracking_uri: Optional[str] = None,
    experiment_name: str = "stock-forecast-mvp4-gnn"
) -> Dict:
    """
    Uczy GNN na snapshotach train/test:
      • train snapshot = ostatnie dostępne punkty ≤ date_split
      • test snapshot  = ostatnie dostępne punkty  > date_split
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Target y (log-return fwd)
    df["y"] = compute_forward_return(df, horizon=Y_CONFIG["horizon"], style=Y_CONFIG["style"])
    if Y_CONFIG["winsorize"]:
        df["y"] = _winsorize(df["y"], Y_CONFIG["winsorize"])

    last_valid_per_company = df[np.isfinite(df["y"].to_numpy())].groupby("company_id")["date"].max()
    if not last_valid_per_company.empty:
        global_last_valid = last_valid_per_company.min()
        # jeśli split jest po global_last_valid, przesuń na global_last_valid
        if pd.Timestamp(date_split) > global_last_valid:
            date_split = str(global_last_valid.date())

    valid_y = np.isfinite(df["y"].to_numpy())

    split_ts = pd.Timestamp(date_split)
    train_mask = (df["date"] <= split_ts) & valid_y
    test_mask  = (df["date"] >  split_ts) & valid_y

    feature_cols = infer_common_feature_cols(df)

    # Graf (na train)
    adj, cids = build_rolling_corr_adj(df.loc[train_mask], window=corr_window, thr=corr_thr)
    if len(cids) == 0:
        raise ValueError("Brak wystarczających danych do zbudowania grafu (train set pusty lub za mały).")

    adj = add_sector_edges(adj, cids, df)
    # dense_to_sparse zwraca krotkę (edge_index, edge_weight)
    edge_index, _ = dense_to_sparse(torch.as_tensor(adj, dtype=torch.float32))
    edge_index = edge_index.long()

    # Snapshots → tensory (osobno dla train/test)
    Xtr, ytr, cids_tr = make_snapshot_tensor(df, feature_cols, train_mask)
    Xte, yte, cids_te = make_snapshot_tensor(df, feature_cols, test_mask)

    if len(cids_tr) == 0 or len(cids_te) == 0:
        raise ValueError("Brak danych po filtracji snapshotów (wszystko NaN/Inf lub brak targetu).")

    cid_pos = {cid: i for i, cid in enumerate(cids)}
    keep_tr = [i for i, cid in enumerate(cids_tr) if cid in cid_pos]
    keep_te = [i for i, cid in enumerate(cids_te) if cid in cid_pos]

    if not keep_tr or not keep_te:
        raise ValueError("Brak wspólnych spółek między grafem a snapshotami train/test.")

    Xtr, ytr = Xtr[keep_tr], ytr[keep_tr]
    Xte, yte = Xte[keep_te], yte[keep_te]
    idx_tr = torch.tensor([cid_pos[cids_tr[i]] for i in keep_tr], dtype=torch.long)
    idx_te = torch.tensor([cid_pos[cids_te[i]] for i in keep_te], dtype=torch.long)

    # X_nodes dla train i test (oddzielnie, aby test nie był zerowy)
    X_nodes_tr = torch.zeros((len(cids), Xtr.shape[1]), dtype=torch.float32)
    X_nodes_te = torch.zeros((len(cids), Xte.shape[1]), dtype=torch.float32)
    X_nodes_tr[idx_tr] = Xtr
    X_nodes_te[idx_te] = Xte

    # === Model z per-company headami ===
    in_dim = X_nodes_tr.shape[1]
    hidden = 64
    model = HeadedGNN(in_dim, hidden, company_ids=cids, kind=model_kind).to(DEVICE)

    X_nodes_tr = X_nodes_tr.to(DEVICE)
    X_nodes_te = X_nodes_te.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    ytr = ytr.to(DEVICE)
    yte = yte.to(DEVICE)
    idx_tr = idx_tr.to(DEVICE)
    idx_te = idx_te.to(DEVICE)

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # MLflow
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    run_name = f"gnn_{model_kind}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name, tags={
        "algo": f"GNN-{model_kind}",
        "heads": "per-company"
    }) as run:
        mlflow.log_params({
            "date_split": date_split,
            "horizon": Y_CONFIG["horizon"],
            "winsorize": Y_CONFIG["winsorize"] or 0.0,
            "corr_window": corr_window,
            "corr_thr": corr_thr,
            "in_dim": in_dim,
            "hidden": hidden,
            "model_kind": model_kind,
            "lr": lr,
            "epochs": epochs,
            "n_nodes": len(cids),
            "n_train_nodes": int(len(idx_tr)),
            "n_test_nodes": int(len(idx_te)),
            "feature_cols": ",".join(feature_cols),
        })

        best_te = np.inf
        best_state = None

        for ep in range(1, epochs + 1):
            # --- train ---
            model.train()
            opt.zero_grad()
            h_tr = model(X_nodes_tr, edge_index)                 # [N, H]
            pred_tr = model.predict_for_indices(h_tr, idx_tr, cids)
            loss = loss_fn(pred_tr, ytr)
            loss.backward()
            opt.step()

            # --- eval ---
            model.eval()
            with torch.no_grad():
                h_te = model(X_nodes_te, edge_index)
                pred_te = model.predict_for_indices(h_te, idx_te, cids)
                te_mse = loss_fn(pred_te, yte).item()
                te_rmse = float(np.sqrt(te_mse))
                te_mae = float(torch.mean(torch.abs(pred_te - yte)).item())
                tr_mse = float(loss.item())
                tr_rmse = float(np.sqrt(tr_mse))
                tr_mae = float(torch.mean(torch.abs(pred_tr - ytr)).item())

            safe_log_metrics({
                "train_mse": tr_mse, "train_rmse": tr_rmse, "train_mae": tr_mae,
                "test_mse": te_mse, "test_rmse": te_rmse, "test_mae": te_mae
            }, step=ep)

            if np.isfinite(te_rmse) and te_rmse < best_te:
                best_te = te_rmse
                best_state = {
                    "state_dict": model.state_dict(),
                    "in_dim": in_dim,
                    "hidden": hidden,
                    "model_kind": model_kind,
                    "cids": cids,
                    "feature_cols": feature_cols,
                }

        # ===== Export per-company =====
        export_dir = Path("artifacts") / "companies"
        export_dir.mkdir(parents=True, exist_ok=True)

        if best_state is None:
            best_state = {
                "state_dict": model.state_dict(),
                "in_dim": in_dim,
                "hidden": hidden,
                "model_kind": model_kind,
                "cids": cids,
                "feature_cols": feature_cols,
            }

        full_state = best_state["state_dict"]
        for cid in cids:
            single = {
                "encoder": {k: v for k, v in full_state.items() if not k.startswith("heads.")},
                "head": full_state[f"heads.{cid}.weight"],
                "head_bias": full_state[f"heads.{cid}.bias"],
                "in_dim": best_state["in_dim"],
                "hidden": best_state["hidden"],
                "model_kind": best_state["model_kind"],
                "company_id": cid,
                "feature_cols": best_state["feature_cols"],
            }
            out_path = export_dir / f"gnn_company_{cid}.pt"
            torch.save(single, out_path)
            mlflow.log_artifact(str(out_path), artifact_path=f"companies/{cid}")

        return {
            "best_test_rmse": float(best_te) if np.isfinite(best_te) else float("inf"),
            "n_nodes": len(cids),
            "n_train_nodes": int(len(idx_tr)),
            "n_test_nodes": int(len(idx_te)),
            "exported_companies": len(cids)
        }

# ======================
# Public API
# ======================
def train_gnn(
    company_ids: List[int],
    *,
    date_split: str = "2023-12-31",
    max_date: str = "2024-12-31",
    corr_window: int = 120,
    corr_thr: float = 0.5,
    model_kind: str = "gcn",    # "gcn" lub "gat"
    lr: float = 1e-3,
    epochs: int = 50,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "stock-forecast-mvp4-gnn",
) -> Dict:
    df = load_panel(company_ids)
    if df.empty:
        raise ValueError("No data for given company_ids")

    df = df[df["date"] <= pd.to_datetime(max_date)]

    return train_one_snapshot(
        df=df,
        date_split=date_split,
        corr_window=corr_window,
        corr_thr=corr_thr,
        lr=lr,
        epochs=epochs,
        model_kind=model_kind,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )

# ======================
# CLI
# ======================
if __name__ == "__main__":
    companies = get_companies_by_group("wig20")
    company_ids = [int(company.id) for company in companies]

    out = train_gnn(
        company_ids=company_ids,
        date_split="2023-12-31",
        max_date="2024-12-31",
        corr_window=120,
        corr_thr=0.5,
        model_kind="gcn",
        lr=1e-3,
        epochs=50,
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="stock-forecast-mvp4-gnn",
    )
    print(out)
