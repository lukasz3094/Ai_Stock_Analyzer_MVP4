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
from lightgbm import LGBMRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from app.branches.prediction_branch.data_loader.features_loader import get_data_for_model_per_sector_v1, get_data_for_model_per_sector_v2, get_data_for_model_per_sector_v3

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

def _sign_hit_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    s_true = np.sign(y_true.values)
    s_pred = np.sign(y_pred.values)
    return float((s_true == s_pred).mean())

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

def train_lgbm_for_sector_v3(
    sector_id: int,
    *,
    date_split: str = "2023-12-31",
    rolling_window_days: Optional[int] = 540,
    selected_features: Optional[List[str]] = None,  # optional whitelist
    tracking_uri: str = None,
    experiment_name: str = "stock-forecast-v3",
    artifacts_dir: str | Path = "artifacts",
    model_registry_name_tpl: str = "stock_model_sector_{sid}_lgbm",
) -> Dict:
    df = get_data_for_model_per_sector_v3(sector_id)
    if df.empty:
        raise ValueError(f"No data for sector {sector_id}")

    df = _clip_rolling_window(df, rolling_window_days)

    # reuse your target creation & split (works for any feature set)
    Xtr, ytr, Xte, yte, used_cols, y_name, train_meta, test_meta = _prepare_xy(
        df, selected_features, date_split
    )

    # align checks
    assert Xtr.index.equals(ytr.index)
    assert Xte.index.equals(yte.index)
    assert test_meta.index.equals(yte.index)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    params = dict(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        objective="regression",
        verbose=-1,
    )
    model = LGBMRegressor(**params)

    pipe = Pipeline([
        # trees don’t need standardization; keep scaler only if some algos require it
        ("model", model),
    ])

    run_name = f"sector_{sector_id}_lgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name, tags={"sector_id": str(sector_id), "algo": "LGBM"}) as run:
        mlflow.log_params({"date_split": date_split, "rolling_window_days": rolling_window_days or 0,
                           "n_features": len(used_cols), **{f"lgbm_{k}": v for k,v in params.items()}})

        # simple time-series CV for reporting (optional)
        cv = timeseries_cv_metrics(Xtr, ytr, n_splits=5, gap=Y_CONFIG["horizon"])
        mlflow.log_metrics({f"br_{k}": v for k, v in cv.items()})  # (labelled; function uses BR internally)

        # Fit with a small validation slice from the tail of train (no leakage)
        split_point = int(len(Xtr) * 0.85)
        Xtr_fit, ytr_fit = Xtr.iloc[:split_point], ytr.iloc[:split_point]
        Xval, yval = Xtr.iloc[split_point:], ytr.iloc[split_point:]

        model.fit(
            Xtr_fit, ytr_fit,
            eval_set=[(Xval, yval)],
            eval_metric="l2",
            callbacks=[]
        )

        pred_tr = model.predict(Xtr)
        pred_te = pd.Series(model.predict(Xte), index=yte.index)

        # index-safe pred_df + price reconstruction plot (same as your V2)
        pred_df = test_meta.join(pd.DataFrame({"y_true": yte, "y_pred": pred_te}), how="inner")

        # base close to reconstruct prices
        base_close = "close" if "close" in test_meta.columns else "close"
        if base_close not in pred_df.columns:
            _df_close = df[["company_id","date","close"]]
            pred_df = pred_df.merge(_df_close, on=["company_id","date"], how="left")
        pred_df["close_base"] = pred_df["close"]
        h = int(Y_CONFIG["horizon"])
        pred_df["date_target"] = pred_df["date"] + pd.to_timedelta(h, unit="D")
        pred_df["close_true_target"] = pred_df["close_base"] * np.exp(pred_df["y_true"].astype(float))
        pred_df["close_pred_target"] = pred_df["close_base"] * np.exp(pred_df["y_pred"].astype(float))

        tr = {"mae": mean_absolute_error(ytr, pred_tr),
              "rmse": np.sqrt(mean_squared_error(ytr, pred_tr)),
              "r2": r2_score(ytr, pred_tr)}
        te = {"mae": mean_absolute_error(yte, pred_te),
              "rmse": np.sqrt(mean_squared_error(yte, pred_te)),
              "r2": r2_score(yte, pred_te)}

        mlflow.log_metrics({f"train_{k}": v for k,v in tr.items()})
        mlflow.log_metrics({f"test_{k}": v for k,v in te.items()})
        mlflow.log_metric("test_hit_rate", _sign_hit_rate(yte, pred_te))
        if len(yte) >= 2:
            mlflow.log_metric("test_ic_pearson_np", float(np.corrcoef(yte.to_numpy(), pred_te.to_numpy())[0,1]))

        # log model
        sig = infer_signature(Xtr, model.predict(Xtr))
        mlflow.sklearn.log_model(pipe, artifact_path="model", signature=sig, input_example=Xtr.head(3))

        reg_name = model_registry_name_tpl.format(sid=sector_id)
        result = mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=reg_name)
        mlflow.set_tag("registered_model_version", result.version)

        # plots (returns; price with targets)
        cid = pred_df["company_id"].value_counts().idxmax()
        sample = pred_df.loc[pred_df["company_id"] == cid].sort_values("date").tail(300)

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(sample["date"], sample["y_true"], label="y_true")
        ax.plot(sample["date"], sample["y_pred"], label="y_pred", alpha=0.8)
        ax.axhline(0, linewidth=1)
        ax.set_title(f"Sector {sector_id} - company_id={cid} (LGBM test)")
        ax.set_ylabel("log-return (fraction)")
        ax.legend(); fig.autofmt_xdate()
        mlflow.log_figure(fig, f"plots/v3_returns_company_{cid}.png"); plt.close(fig)

        hist = df.loc[df["company_id"] == cid, ["date","close"]].dropna().sort_values("date")
        pts = pred_df.loc[pred_df["company_id"] == cid, ["date_target","close_pred_target","close_true_target"]].dropna()
        fig2, ax2 = plt.subplots(figsize=(11,4.5))
        ax2.plot(hist["date"], hist["close"], label="historical close")
        ax2.scatter(pts["date_target"], pts["close_true_target"], s=12, label="true close (target)")
        ax2.scatter(pts["date_target"], pts["close_pred_target"], s=12, label="predicted close (target)")
        ax2.set_title(f"Sector {sector_id} - company_id={cid}: historical & predicted close (V3)")
        ax2.set_ylabel("price"); ax2.legend(); fig2.autofmt_xdate()
        mlflow.log_figure(fig2, f"plots/v3_price_company_{cid}.png"); plt.close(fig2)

        return {"sector_id": sector_id, "model_name": reg_name, "version": result.version,
                "train_metrics": tr, "test_metrics": te, "x_used": used_cols}

if __name__ == "__main__":
    res = train_lgbm_for_sector_v3(
        sector_id=4,
        date_split="2023-12-31",
        rolling_window_days=540,  # lub None dla pełnej historii
        selected_features=None,
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="stock-forecast-mvp4",
    )