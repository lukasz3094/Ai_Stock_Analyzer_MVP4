from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error
)

def strip_comment_keys(obj):
    if isinstance(obj, dict):
        return {k: strip_comment_keys(v) for k, v in obj.items() if not k.endswith("_comment")}
    if isinstance(obj, list):
        return [strip_comment_keys(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(strip_comment_keys(x) for x in obj)
    return obj

def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def require(dic: dict, key: str, section: str):
    if key not in dic:
        raise KeyError(f"Missing required key '{key}' in section '{section}'")
    return dic[key]

def require_keys(dic: dict, keys: list[str], section: str):
    for k in keys:
        if k not in dic:
            raise KeyError(f"Missing required key '{k}' in section '{section}'")

@dataclass
class SplitResult:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    train_end: pd.Timestamp
    company_train: pd.Series
    company_valid: pd.Series 

def time_series_split_with_gap(
    frame: pd.DataFrame,
    feature_cols: list[str],
    y: pd.Series,
    valid_size: float,
    min_train_size: int,
    purge_gap: int,
) -> SplitResult:
    n = len(frame)
    split_idx = max(int(n * (1 - valid_size)), min_train_size)
    X_train = frame.iloc[: split_idx - purge_gap][feature_cols]
    y_train = y.iloc[: split_idx - purge_gap]
    X_valid = frame.iloc[split_idx:][feature_cols]
    y_valid = y.iloc[split_idx:]

    train_end = pd.to_datetime(frame.iloc[split_idx - 1]["date"])

    # meta: company_id do ważenia
    if "company_id" not in frame.columns:
        raise ValueError("Brak kolumny 'company_id' w frame – potrzebna do balance_companies=True.")
    company_train = frame.iloc[: split_idx - purge_gap]["company_id"].reset_index(drop=True)
    company_valid = frame.iloc[split_idx:]["company_id"].reset_index(drop=True)

    return SplitResult(X_train, y_train, X_valid, y_valid, train_end, company_train, company_valid)

def build_time_decay_weights(length: int, half_life_days: int) -> np.ndarray:
    decay = np.log(2) / half_life_days
    t = np.arange(length)
    w = np.exp(decay * t)
    return w / w.mean()

def train_xgb(
    frame: pd.DataFrame,
    feature_cols: list[str],
    y: pd.Series,
    cfg_train: dict,
    xgb_params: dict,
    task: str,
):
    require_keys(cfg_train, ["valid_size", "min_train_size", "purge_gap",
                             "num_boost_round", "early_stopping_rounds",
                             "seed", "use_time_decay", "half_life_days",
                             "compute_scale_pos_weight"], "train")

    split = time_series_split_with_gap(
        frame=frame,
        feature_cols=feature_cols,
        y=y,
        valid_size=cfg_train["valid_size"],
        min_train_size=cfg_train["min_train_size"],
        purge_gap=cfg_train["purge_gap"],
    )

    weights = None
    if cfg_train.get("use_time_decay", False) or cfg_train.get("balance_companies", False):
        td = np.ones(len(split.X_train), dtype=float)
        if cfg_train.get("use_time_decay", False):
            td = build_time_decay_weights(len(split.X_train), int(cfg_train["half_life_days"]))

        bc = np.ones(len(split.X_train), dtype=float)
        if cfg_train.get("balance_companies", False):
            counts = split.company_train.value_counts()
            inv = split.company_train.map(1.0 / counts)
            bc = inv.to_numpy(dtype=float)

        weights = td * bc
        weights *= (len(weights) / weights.sum())

    dtrain = xgb.DMatrix(split.X_train, label=split.y_train.values,
                         feature_names=feature_cols, weight=weights)
    dvalid = xgb.DMatrix(split.X_valid, label=split.y_valid.values,
                         feature_names=feature_cols)

    params = xgb_params.copy()
    if task == "clf" and cfg_train["compute_scale_pos_weight"]:
        pos = float(split.y_train.sum())
        neg = float(len(split.y_train) - pos)
        spw = max(1.0, neg / max(1.0, pos))
        print(f"[INFO] pos={pos}, neg={neg}, scale_pos_weight={round(spw,3)}")
        params["scale_pos_weight"] = spw

    evals_result = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        evals_result=evals_result,
        num_boost_round=cfg_train["num_boost_round"],
        early_stopping_rounds=cfg_train["early_stopping_rounds"],
        verbose_eval=False,
    )
    best_iter = booster.best_iteration or 0
    print(f"[INFO] Best iteration: {best_iter}")
    return booster, split

def evaluate_model(booster, split: SplitResult, feature_cols: list[str], cfg_eval: dict, task: str) -> dict:
    require_keys(cfg_eval, ["classification_threshold"], "eval")
    dvalid = xgb.DMatrix(split.X_valid, feature_names=feature_cols)
    y_true = split.y_valid.values
    y_pred = booster.predict(dvalid)

    if task == "clf":
        thr = cfg_eval["classification_threshold"]
        y_bin = (y_pred >= thr).astype(int)
        metrics = {
            "auc": float(roc_auc_score(y_true, y_pred)),
            "logloss": float(log_loss(y_true, y_pred)),
            "accuracy": float(accuracy_score(y_true, y_bin)),
            "precision": float(precision_score(y_true, y_bin, zero_division=0)),
            "recall": float(recall_score(y_true, y_bin, zero_division=0)),
            "f1": float(f1_score(y_true, y_bin, zero_division=0)),
        }
    else:
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        hit = float((np.sign(y_true) == np.sign(y_pred)).mean())
        metrics = {"rmse": rmse, "mae": mae, "hit_rate": hit}

    print("[VALID METRICS]", metrics)
    return metrics

def feature_importance_report(booster, feature_cols: list[str], cfg_report: dict) -> pd.DataFrame:
    require_keys(cfg_report, ["top_features_n"], "report")
    score = booster.get_score(importance_type="gain")
    df = (
        pd.DataFrame([(f, score.get(f, 0)) for f in feature_cols], columns=["feature", "gain"])
        .sort_values("gain", ascending=False)
        .head(cfg_report["top_features_n"])
    )
    print("\n[TOP FEATURES BY GAIN]")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    try:
        cfg_path = Path(__file__).resolve().parent / "xgb_config.json"
        cfg = load_config(cfg_path)
        cfg = strip_comment_keys(cfg)

        require_keys(cfg, ["data", "features", "train", "xgb_params", "eval", "report"], "root")

        data_cfg = cfg["data"]
        require_keys(data_cfg, ["date_end"], "data")

        from app.services.getters.companies_getter import get_companies_by_sector_id

        companies = get_companies_by_sector_id(34)
        companies_ids = [c.id for c in companies]

        df_full = pd.DataFrame()

        from app.branches.prediction_branch.data_loader.features_loader import get_data_for_model_per_company
        for company_id in companies_ids:
            df = get_data_for_model_per_company(company_id=company_id)
            df = df[pd.to_datetime(df["date"]) <= pd.Timestamp(data_cfg["date_end"])]
            df = df.sort_values("date").reset_index(drop=True)
            df_full = pd.concat([df_full, df], ignore_index=True)

        feat_cfg = cfg["features"]
        require_keys(feat_cfg, ["n_lags", "horizon", "target_style", "winsorize_p",
                                "drop_ohlcv", "add_indicators", "short_horizon_pack"], "features")

        from ml.train.xgboost.prepare_features_xgboost import prepare_features
        frame, feature_cols, y, xgb_default_params, _ = prepare_features(
            df_full,
            n_lags=int(feat_cfg["n_lags"]),
            horizon=int(feat_cfg["horizon"]),
            target_style=str(feat_cfg["target_style"]),
            winsorize_p=None if feat_cfg["winsorize_p"] is None else float(feat_cfg["winsorize_p"]),
            drop_ohlcv=bool(feat_cfg["drop_ohlcv"]),
            add_indicators=bool(feat_cfg["add_indicators"]),
            short_horizon_pack=bool(feat_cfg["short_horizon_pack"]),
            remove_constant_features=True,
            known_lag_sessions=int(feat_cfg["known_lag_sessions"]),
            add_company_id_feature=bool(feat_cfg["add_company_id_feature"]),
            add_sector_id_feature=bool(feat_cfg["add_sector_id_feature"])
        )

        y = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid_idx = y.dropna().index
        frame = frame.loc[valid_idx].reset_index(drop=True)

        if "company_id" not in frame.columns and "company_id" in df_full.columns:
            frame = frame.join(df_full.loc[frame.index, ["company_id"]], how="left")
            if "company_id" not in frame.columns:
                raise RuntimeError("Nie udało się przywrócić 'company_id' do frame.")

        y = y.loc[valid_idx].reset_index(drop=True)

        train_cfg = cfg["train"]
        task = str(require(train_cfg, "task", "train"))

        booster, split = train_xgb(
            frame=frame,
            feature_cols=feature_cols,
            y=y,
            cfg_train=train_cfg,
            xgb_params=cfg["xgb_params"],
            task=task,
        )

        metrics = evaluate_model(booster, split, feature_cols, cfg["eval"], task=task)
        feature_importance_report(booster, feature_cols, cfg["report"])

        io_cfg = cfg.get("io", {})
        if io_cfg.get("save_model", False):
            out_path = Path(io_cfg["model_path"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(out_path.as_posix())
            print(f"[INFO] Model saved to: {out_path}")

        print("\n=== TRAINING COMPLETE ===")
        print(metrics)

    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
