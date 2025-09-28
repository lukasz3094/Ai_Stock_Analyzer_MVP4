from __future__ import annotations
import os, json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from app.branches.prediction_branch.data_loader.features_loader import get_data_for_model_per_sector_v1, get_data_for_model_per_sector_v2

import os
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

ALWAYS_KEEP = ("date", "company_id", "close")

Y_CONFIG = {
    "horizon": 5,
    "style": "log",
    "use_adjusted": True,
    "winsorize": 0.005,
    "scale": "fraction",
    "neutralize_sector": True,
}


# -------- helpers --------
def timeseries_cv_metrics(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, gap: int = None) -> dict:
    if gap is None:
        gap = Y_CONFIG.get("horizon", 1)
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    maes, rmses = [], []
    for tr_idx, va_idx in tscv.split(X):
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", BayesianRidge())
        ])
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = pipe.predict(X.iloc[va_idx])
        maes.append(mean_absolute_error(y.iloc[va_idx], p))
        rmses.append(np.sqrt(mean_squared_error(y.iloc[va_idx], p)))
    return {
        "cv_mae_mean": float(np.mean(maes)),
        "cv_mae_std": float(np.std(maes)),
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_rmse_std": float(np.std(rmses)),
    }

def _to_bp(x: float) -> float:
    return 1e4 * x  # 1 bp = 0.0001

def _sign_hit_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    s_true = np.sign(y_true.values)
    s_pred = np.sign(y_pred.values)
    return float((s_true == s_pred).mean())

def _baselines(y_train: pd.Series, train_meta: pd.DataFrame,
               y_test: pd.Series,  test_meta: pd.DataFrame) -> dict:
    zero_train = np.zeros_like(y_train)
    zero_test  = np.zeros_like(y_test)

    h = int(Y_CONFIG.get("horizon", 1))
    prev_non_overlap = (test_meta
                        .sort_values(["company_id","date"])
                        .groupby("company_id")["y"]
                        .shift(h))
    prev_non_overlap = prev_non_overlap.reindex(y_test.index).fillna(0.0)

    out = {}
    out["baseline0_train_mae"] = mean_absolute_error(y_train, zero_train)
    out["baseline0_test_mae"]  = mean_absolute_error(y_test,  zero_test)
    out["baseline_prev_nonoverlap_test_mae"] = mean_absolute_error(y_test, prev_non_overlap)
    out["baseline0_test_hit_rate"] = _sign_hit_rate(y_test, pd.Series(zero_test, index=y_test.index))
    out["baseline_prev_nonoverlap_test_hit_rate"] = _sign_hit_rate(y_test, prev_non_overlap)
    return out


def _neutralize_sector_mean(df: pd.DataFrame) -> pd.DataFrame:
    adj = df.copy()
    daily_mean = adj.groupby("date")["y"].transform("mean")
    adj["y"] = adj["y"] - daily_mean
    return adj

def _select_price_column(df: pd.DataFrame, use_adjusted: bool) -> str:
    if use_adjusted:
        for col in ("adj_close", "adjusted_close", "close_adj"):
            if col in df.columns:
                return col
    return "close"

def _compute_forward_return(
    df: pd.DataFrame,
    company_col: str,
    price_col: str,
    horizon: int,
    style: str = "log",
) -> pd.Series:
    df = df.sort_values([company_col, "date"]).copy()
    s = df[price_col].astype(float)
    fwd = df.groupby(company_col)[price_col].shift(-horizon) / s
    if style == "log":
        y = np.log(fwd)
    else:
        y = fwd - 1.0
    return y

def _winsorize(s: pd.Series, alpha: float) -> pd.Series:
    lo = s.quantile(alpha)
    hi = s.quantile(1 - alpha)
    return s.clip(lower=lo, upper=hi)

def _make_params_dict(
    sector_id: int,
    date_split: str,
    rolling_window_days: Optional[int],
    selected_features: Optional[List[str]],
    used_x_columns: List[str],
    model_params: Dict,
    scaler_params: Dict,
    df_meta: Dict,
    y_config: Dict,
    y_name: str
) -> Dict:
    return {
        "sector_id": sector_id,
        "date_split": date_split,
        "rolling_window_days": rolling_window_days,
        "feature_source": "selected_features_boruta",
        "always_keep": list(ALWAYS_KEEP),
        "selected_features_input": selected_features or [],
        "x_columns_used": used_x_columns,
        "target": "next_day_close_return_pct_change",
        "model": {"type": "BayesianRidge", "params": model_params},
        "scaler": {"type": "StandardScaler", "params": scaler_params},
        "dataset": df_meta,  # n_rows, date_min, date_max
        "target_name": y_name,
        "target_config": y_config,
    }

def _save_params_json(params: Dict, sector_id: int, out_dir: str | Path = "artifacts") -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"params_sector_{sector_id}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    return path

def _clip_rolling_window(df: pd.DataFrame, days: Optional[int]) -> pd.DataFrame:
    if not days:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=False).dt.tz_localize(None)
    end = df["date"].max()
    start = end - pd.Timedelta(days=days)
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()

def _prepare_xy(
    df: pd.DataFrame,
    selected_features: Optional[List[str]],
    date_split: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], str]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=False).dt.tz_localize(None)

    trade_mask = df["close"].notna()
    if "volume" in df.columns:
        trade_mask &= df["volume"].notna() & (df["volume"].astype(float) > 0)
    df = df.loc[trade_mask].copy()

    price_col = _select_price_column(df, use_adjusted=Y_CONFIG["use_adjusted"])
    if price_col not in df.columns:
        raise ValueError(f"Required price column '{price_col}' not found.")

    y = _compute_forward_return(
        df=df,
        company_col="company_id",
        price_col=price_col,
        horizon=Y_CONFIG["horizon"],
        style=Y_CONFIG["style"],
    )

    if Y_CONFIG["winsorize"]:
        y = _winsorize(y, float(Y_CONFIG["winsorize"]))

    y_name = f"y_{Y_CONFIG['style']}_h{Y_CONFIG['horizon']}_{Y_CONFIG['scale']}"

    df["y"] = y

    if Y_CONFIG["neutralize_sector"]:
        df = _neutralize_sector_mean(df)

    if Y_CONFIG["winsorize"]:
        alpha = float(Y_CONFIG["winsorize"])
        lo = df["y"].quantile(alpha)
        hi = df["y"].quantile(1 - alpha)
        df["y"] = df["y"].clip(lo, hi)

    blacklist = set(ALWAYS_KEEP) | {"y", "id", "sector_id", "created_at"}
    if selected_features:
        x_cols = [c for c in selected_features if c in df.columns and c not in {"date", "company_id"}]
    else:
        x_cols = [c for c in df.columns if c not in blacklist]

    date_split = pd.to_datetime(date_split).tz_localize(None)
    train = df[df["date"] <= date_split].dropna(subset=["y"]).copy()
    test  = df[df["date"] >  date_split].dropna(subset=["y"]).copy()

    X_train = train[x_cols].astype(float).dropna(axis=0, how="any")
    y_train = train.loc[X_train.index, "y"].astype(float)

    X_test = test[x_cols].astype(float).dropna(axis=0, how="any")
    y_test = test.loc[X_test.index, "y"].astype(float)

    if X_train.empty or X_test.empty:
        raise ValueError(
            f"Empty X after filtering. Used cols: {x_cols}. "
            f"Train rows: {len(X_train)}, Test rows: {len(X_test)}."
        )
    
    train_meta = train.loc[X_train.index, ["company_id","date","y"]].copy()
    test_meta  = test.loc[X_test.index,  ["company_id","date","y"]].copy()
    return X_train, y_train, X_test, y_test, x_cols, y_name, train_meta, test_meta

def _evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    pred = model.predict(X)

    try:
        rmse = mean_squared_error(y, pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y, pred))
    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": rmse,
        "r2": r2_score(y, pred),
    }

# -------- public --------

# V1 with calculations
def train_bayesianridge_for_sector_v1(
    sector_id: int,
    *,
    date_split: str = "2023-12-31",
    rolling_window_days: Optional[int] = None,
    selected_features: Optional[List[str]] = None,
    tracking_uri: str = None,
    experiment_name: str = "stock-forecast-mvp4",
    artifacts_dir: str | Path = "artifacts",
    model_registry_name_tpl: str = "stock_model_sector_{sid}_bayes",
) -> Dict:
    df = get_data_for_model_per_sector_v1(sector_id)
    if df.empty:
        raise ValueError(f"No data for sector {sector_id}")

    df = _clip_rolling_window(df, rolling_window_days)

    Xtr, ytr, Xte, yte, used_cols, y_name, train_meta, test_meta = _prepare_xy(df, selected_features, date_split)

    model_params = {}
    scaler_params = {"with_mean": True, "with_std": True}
    df_meta = {
        "n_rows_total": int(len(df)),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
    }

    params_dict = _make_params_dict(
        sector_id=sector_id,
        date_split=date_split,
        rolling_window_days=rolling_window_days,
        selected_features=selected_features,
        used_x_columns=used_cols,
        model_params=model_params,
        scaler_params=scaler_params,
        df_meta=df_meta,
        y_config=Y_CONFIG,
        y_name=y_name,
    )

    params_dict["target_name"] = f"y_{Y_CONFIG['style']}_h{Y_CONFIG['horizon']}_{Y_CONFIG['scale']}"
    params_dict["target_config"] = Y_CONFIG

    params_path = _save_params_json(params_dict, sector_id, artifacts_dir)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    pipe = Pipeline([
        ("scaler", StandardScaler(**scaler_params)),
        ("model", BayesianRidge(**model_params)),
    ])

    run_name = f"sector_{sector_id}_bayes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name, tags={"sector_id": str(sector_id), "algo": "BayesianRidge"}) as run:
        mlflow.log_params({
            "date_split": date_split,
            "rolling_window_days": rolling_window_days or 0,
            "n_features": len(used_cols),
        })
        mlflow.log_text(json.dumps(used_cols, ensure_ascii=False, indent=2), "used_features.json")
        mlflow.log_artifact(str(params_path))

        cv = timeseries_cv_metrics(Xtr, ytr, n_splits=5, gap=Y_CONFIG["horizon"])
        mlflow.log_metrics(cv)

        pipe.fit(Xtr, ytr)

        pred_tr = pipe.predict(Xtr)
        pred_te = pipe.predict(Xte)

        pred_df = test_meta.copy()
        pred_df["y_true"] = yte.values
        pred_df["y_pred"] = pred_te

        tr = {
            "mae": mean_absolute_error(ytr, pred_tr),
            "rmse": np.sqrt(mean_squared_error(ytr, pred_tr)),
            "r2": r2_score(ytr, pred_tr),
        }
        te = {
            "mae": mean_absolute_error(yte, pred_te),
            "rmse": np.sqrt(mean_squared_error(yte, pred_te)),
            "r2": r2_score(yte, pred_te),
        }

        tr_bp = {f"{k}_bp": _to_bp(v) for k, v in tr.items() if k in ("mae","rmse")}
        te_bp = {f"{k}_bp": _to_bp(v) for k, v in te.items() if k in ("mae","rmse")}

        metrics_extra = {
            "train_hit_rate": _sign_hit_rate(ytr, pd.Series(pred_tr, index=ytr.index)),
            "test_hit_rate":  _sign_hit_rate(yte, pd.Series(pred_te, index=yte.index)),
        }
        if len(yte) >= 2:
            try:
                mlflow.log_metric("test_ic_pearson_np", float(np.corrcoef(yte.to_numpy(), pred_te)[0, 1]))
            except Exception:
                pass

        base = _baselines(ytr, train_meta, yte, test_meta)
        mlflow.log_metrics(base)

        mlflow.log_metrics({f"train_{k}": v for k, v in tr.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in te.items()})
        mlflow.log_metrics({f"train_{k}": v for k, v in tr_bp.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in te_bp.items()})
        mlflow.log_metrics(metrics_extra)

        signature = infer_signature(Xtr, pipe.predict(Xtr))
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            signature=signature,
            input_example=Xtr.head(3)
        )

        reg_name = model_registry_name_tpl.format(sid=sector_id)
        result = mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=reg_name)
        mlflow.set_tag("registered_model_version", result.version)

        cid = pred_df.groupby("company_id").size().idxmax()
        sample = pred_df[pred_df["company_id"] == cid].sort_values("date").tail(300)

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(sample["date"], sample["y_true"], label="y_true")
        ax.plot(sample["date"], sample["y_pred"], label="y_pred", alpha=0.8)
        if "y_pred_cal" in sample:
            ax.plot(sample["date"], sample["y_pred_cal"], label="y_pred_cal", alpha=0.8, linestyle="--")
        ax.axhline(0, linewidth=1)
        ax.set_title(f"Sector {sector_id} - company_id={cid} (test)")
        ax.set_ylabel("log-return (fraction)")
        ax.legend()
        fig.autofmt_xdate()
        mlflow.log_figure(fig, f"plots/series_company_{cid}.png")
        plt.close(fig)

        return {
            "sector_id": sector_id,
            "model_name": reg_name,
            "version": result.version,
            "train_metrics": tr,
            "test_metrics": te,
            "params_json": str(params_path),
            "x_used": used_cols,
        }

# V2 without calculations
def train_bayesianridge_for_sector_v2(
    sector_id: int,
    *,
    date_split: str = "2023-12-31",
    rolling_window_days: Optional[int] = None,
    selected_features: Optional[List[str]] = None,
    tracking_uri: str = None,
    experiment_name: str = "stock-forecast-mvp4",
    artifacts_dir: str | Path = "artifacts",
    model_registry_name_tpl: str = "stock_model_sector_{sid}_bayes",
) -> Dict:
    # ----- data -----
    df = get_data_for_model_per_sector_v2(sector_id)
    if df.empty:
        raise ValueError(f"No data for sector {sector_id}")

    df = _clip_rolling_window(df, rolling_window_days)

    Xtr, ytr, Xte, yte, used_cols, y_name, train_meta, test_meta = _prepare_xy(
        df, selected_features, date_split
    )

    # ----- alignment checks -----
    if not Xtr.index.equals(ytr.index):
        raise ValueError("Train X and y indices are misaligned")
    if not Xte.index.equals(yte.index):
        raise ValueError("Test X and y indices are misaligned")
    if not test_meta.index.equals(yte.index):
        raise ValueError("test_meta must share the same index as yte")

    # ensure datetime
    if "date" in test_meta:
        test_meta = test_meta.copy()
        test_meta["date"] = pd.to_datetime(test_meta["date"], errors="coerce")

    # ----- params / metadata -----
    model_params = {}
    scaler_params = {"with_mean": True, "with_std": True}
    df_meta = {
        "n_rows_total": int(len(df)),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
    }

    params_dict = _make_params_dict(
        sector_id=sector_id,
        date_split=date_split,
        rolling_window_days=rolling_window_days,
        selected_features=selected_features,
        used_x_columns=used_cols,
        model_params=model_params,
        scaler_params=scaler_params,
        df_meta=df_meta,
        y_config=Y_CONFIG,
        y_name=y_name,
    )
    params_dict["target_name"] = f"y_{Y_CONFIG['style']}_h{Y_CONFIG['horizon']}_{Y_CONFIG['scale']}"
    params_dict["target_config"] = Y_CONFIG
    params_path = _save_params_json(params_dict, sector_id, artifacts_dir)

    # ----- MLflow -----
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    pipe = Pipeline([
        ("scaler", StandardScaler(**scaler_params)),
        ("model", BayesianRidge(**model_params)),
    ])

    run_name = f"sector_{sector_id}_bayes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name, tags={"sector_id": str(sector_id), "algo": "BayesianRidge"}) as run:
        mlflow.log_params({
            "date_split": date_split,
            "rolling_window_days": rolling_window_days or 0,
            "n_features": len(used_cols),
        })
        mlflow.log_text(json.dumps(used_cols, ensure_ascii=False, indent=2), "used_features.json")
        mlflow.log_artifact(str(params_path))

        cv = timeseries_cv_metrics(Xtr, ytr, n_splits=5, gap=Y_CONFIG["horizon"])
        mlflow.log_metrics(cv)

        # ----- fit & predict -----
        pipe.fit(Xtr, ytr)
        pred_tr = pipe.predict(Xtr)
        pred_te = pd.Series(pipe.predict(Xte), index=yte.index)  # keep index!

        # ----- build pred_df by index -----
        pred_df = test_meta.join(
            pd.DataFrame({"y_true": yte, "y_pred": pred_te}),
            how="inner"
        )

        # If base close at feature time isn't in meta, merge it from df
        close_col_candidates = [c for c in ["close", "adj_close", "price_close"] if c in test_meta.columns]
        if close_col_candidates:
            base_close_col = close_col_candidates[0]
            pred_df["close_base"] = pred_df[base_close_col]
        else:
            # merge base close from original df on (company_id, date)
            _df_close = (df[["company_id", "date"] + [c for c in ["close","adj_close","price_close"] if c in df.columns]]
                         .rename(columns={"adj_close": "close"}))
            pred_df = pred_df.merge(_df_close, on=["company_id","date"], how="left", suffixes=("",""))
            if "close" not in pred_df:
                raise ValueError("Could not find a close/adj_close column in df or test_meta.")
            pred_df["close_base"] = pred_df["close"]

        # Sort / dedup for clean plotting
        sort_cols = [c for c in ["company_id","date"] if c in pred_df.columns]
        if sort_cols:
            pred_df = pred_df.sort_values(sort_cols).drop_duplicates(sort_cols, keep="last")

        # ----- price-level targets on the ACTUAL target date -----
        # target date = feature date + horizon (days)
        horizon_days = int(Y_CONFIG["horizon"])
        pred_df["date_target"] = pred_df["date"] + pd.to_timedelta(horizon_days, unit="D")

        # y is log-return (fraction): P_{t+h} = P_t * exp(y)
        pred_df["close_true_target"] = pred_df["close_base"] * np.exp(pred_df["y_true"].astype(float))
        pred_df["close_pred_target"] = pred_df["close_base"] * np.exp(pred_df["y_pred"].astype(float))

        # ----- metrics -----
        tr = {
            "mae": mean_absolute_error(ytr, pred_tr),
            "rmse": np.sqrt(mean_squared_error(ytr, pred_tr)),
            "r2": r2_score(ytr, pred_tr),
        }
        te = {
            "mae": mean_absolute_error(yte, pred_te),
            "rmse": np.sqrt(mean_squared_error(yte, pred_te)),
            "r2": r2_score(yte, pred_te),
        }
        tr_bp = {f"{k}_bp": _to_bp(v) for k, v in tr.items() if k in ("mae", "rmse")}
        te_bp = {f"{k}_bp": _to_bp(v) for k, v in te.items() if k in ("mae", "rmse")}
        metrics_extra = {
            "train_hit_rate": _sign_hit_rate(ytr, pd.Series(pred_tr, index=ytr.index)),
            "test_hit_rate":  _sign_hit_rate(yte, pred_te),
        }
        if len(yte) >= 2:
            try:
                mlflow.log_metric("test_ic_pearson_np",
                                  float(np.corrcoef(yte.to_numpy(), pred_te.to_numpy())[0, 1]))
            except Exception:
                pass

        base = _baselines(ytr, train_meta, yte, test_meta)
        mlflow.log_metrics(base)
        mlflow.log_metrics({f"train_{k}": v for k, v in tr.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in te.items()})
        mlflow.log_metrics({f"train_{k}": v for k, v in tr_bp.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in te_bp.items()})
        mlflow.log_metrics(metrics_extra)

        mlflow.log_metric("yte_std", float(yte.std()))
        mlflow.log_metric("pred_te_std", float(pred_te.std()))
        mlflow.log_text(pred_df[["company_id","date","date_target","close_base","y_true","y_pred",
                                 "close_true_target","close_pred_target"]]
                        .tail(12).to_csv(index=False),
                        "sample_pred_tail_prices.csv")

        # ----- model logging -----
        signature = infer_signature(Xtr, pipe.predict(Xtr))
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            signature=signature,
            input_example=Xtr.head(3)
        )

        reg_name = model_registry_name_tpl.format(sid=sector_id)
        result = mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=reg_name)
        mlflow.set_tag("registered_model_version", result.version)

        # ----- plot 1: log-returns on feature date (your original) -----
        if "company_id" in pred_df.columns:
            cid = pred_df["company_id"].value_counts().idxmax()
            sample = pred_df.loc[pred_df["company_id"] == cid].sort_values("date").tail(300)
        else:
            cid = None
            sample = pred_df.sort_values("date").tail(300)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sample["date"], sample["y_true"], label="y_true")
        ax.plot(sample["date"], sample["y_pred"], label="y_pred", alpha=0.8)
        ax.axhline(0, linewidth=1)
        ax.set_title(f"Sector {sector_id}" + (f" - company_id={cid}" if cid is not None else "") + " (test)")
        ax.set_ylabel("log-return (fraction)")
        ax.legend()
        fig.autofmt_xdate()
        mlflow.log_figure(fig, f"plots/series_company_{sector_id}_{cid or 'all'}_returns.png")
        plt.close(fig)

        # ----- plot 2: HISTORICAL CLOSE + predicted/true target close on TARGET date -----
        # full historical for this company (if available)
        if cid is not None and "close" in df.columns:
            hist = df.loc[df["company_id"] == cid, ["date", "close"]].dropna().sort_values("date")

            # points (targets) on date_target
            pts = pred_df.loc[pred_df["company_id"] == cid,
                              ["date_target","close_pred_target","close_true_target"]].dropna()

            fig2, ax2 = plt.subplots(figsize=(11, 4.5))
            ax2.plot(hist["date"], hist["close"], label="historical close")
            # overlay true (reconstructed) and predicted closes at the TARGET date
            ax2.scatter(pts["date_target"], pts["close_true_target"], s=12, label="true close (target date)")
            ax2.scatter(pts["date_target"], pts["close_pred_target"], s=12, label="predicted close (target date)")
            ax2.set_title(f"Sector {sector_id} - company_id={cid}: historical & predicted close")
            ax2.set_ylabel("price")
            ax2.legend()
            fig2.autofmt_xdate()
            mlflow.log_figure(fig2, f"plots/series_company_{sector_id}_{cid}_close_with_preds.png")
            plt.close(fig2)

        return {
            "sector_id": sector_id,
            "model_name": reg_name,
            "version": result.version,
            "train_metrics": tr,
            "test_metrics": te,
            "params_json": str(params_path),
            "x_used": used_cols,
        }

if __name__ == "__main__":
    res = train_bayesianridge_for_sector_v2(
        sector_id=4,
        date_split="2023-12-31",
        rolling_window_days=540,  # lub None dla pełnej historii
        selected_features=None,
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="stock-forecast-mvp4",
    )
