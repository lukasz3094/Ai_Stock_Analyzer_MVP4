import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

# ---- config ----
DROP_OHLCV = False
FUND = {"revenue", "operating_profit", "gross_profit", "net_profit", "ebitda"}
MACRO = {"gdp", "cpi", "unemployment_rate", "interest_rate", "exchange_rate_eur", "exchange_rate_usd"}
OHLCV = {"open", "high", "low", "volume"}

def _pct_change_safe(s: pd.Series, periods: int) -> pd.Series:
    out = s.pct_change(periods=periods)
    return out.replace([np.inf, -np.inf], np.nan)


def prepare_features(df: pd.DataFrame, n_lags: int = 10) -> tuple[pd.DataFrame, list[str]]:
    """
    Expects df with at least ['date','close'] and optionally FUND/MACRO columns.
    Returns (feature_frame, feature_cols). Works on GPW session rows only.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    if DROP_OHLCV:
        df = df.drop(columns=[c for c in OHLCV if c in df.columns], errors="ignore")

    # target: daily log-return
    df["r"] = np.log(df["close"]).diff()

    # price-based features
    for i in range(1, n_lags + 1):
        df[f"r_lag_{i}"] = df["r"].shift(i)
    df["r_ma_5"]  = df["r"].rolling(5).mean()
    df["r_ma_10"] = df["r"].rolling(10).mean()
    df["r_std_5"] = df["r"].rolling(5).std()
    df["r_std_10"] = df["r"].rolling(10).std()

    # exogenous (use yesterday's info to avoid look-ahead)
    present_fund  = sorted(list(FUND.intersection(df.columns)))
    present_macro = sorted(list(MACRO.intersection(df.columns)))
    exog_cols_created = []
    for col in present_fund + present_macro:
        s = pd.to_numeric(df[col], errors="coerce")
        df[f"{col}_lag1"] = s.shift(1)
        df[f"{col}_chg_21d"] = _pct_change_safe(s.shift(1), 21)
        df[f"{col}_chg_63d"] = _pct_change_safe(s.shift(1), 63)
        exog_cols_created += [f"{col}_lag1", f"{col}_chg_21d", f"{col}_chg_63d"]

    # calendar features on session dates
    dow = df["date"].dt.weekday
    df["sin_dow"], df["cos_dow"] = np.sin(2*np.pi*dow/5), np.cos(2*np.pi*dow/5)
    per = df["date"].dt.to_period("M")
    pos = per.groupby(per).cumcount() + 1
    cnt = per.value_counts().reindex(per).to_numpy()
    mp = pos / cnt
    df["sin_mp"], df["cos_mp"] = np.sin(2*np.pi*mp), np.cos(2*np.pi*mp)

    # finalize features
    price_feats = [c for c in df.columns if c.startswith(("r_lag_","r_ma_","r_std_"))]
    cal_feats   = ["sin_dow","cos_dow","sin_mp","cos_mp"]
    exog_feats  = [c for c in exog_cols_created if c in df.columns]
    feature_cols = price_feats + cal_feats + exog_feats

    df = df.dropna(subset=["r"] + feature_cols).reset_index(drop=True)
    df = df.drop(columns=[c for c in ("id","company_id","sector_id","created_at") if c in df.columns], errors="ignore")

    return df, feature_cols
