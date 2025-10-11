import numpy as np
import pandas as pd
from typing import Literal, Tuple, Dict, List, Optional

FUND = {"revenue", "operating_profit", "gross_profit", "net_profit", "ebitda"}
MACRO = {"gdp", "cpi", "unemployment_rate", "interest_rate",
         "exchange_rate_eur", "exchange_rate_usd"}
OHLCV = {"open", "high", "low", "volume"}

def _pct_change_safe(s: pd.Series, periods: int) -> pd.Series:
    out = s.pct_change(periods=periods)
    return out.replace([np.inf, -np.inf], np.nan)

def _drop_constant_cols(df: pd.DataFrame, exclude: List[str] | None = None) -> pd.DataFrame:
    exclude = set(exclude or [])
    nunique = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore") \
                .nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    return df.drop(columns=const_cols, errors="ignore")


def _winsorize_series(s: pd.Series, p: float) -> pd.Series:
    if p is None or p <= 0:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)

def make_target_grouped(
    df: pd.DataFrame,
    horizon: int,
    style: Literal["log", "direction"],
    winsorize_p: Optional[float]
) -> pd.Series:
    def _one(g: pd.DataFrame, company_id) -> pd.Series:
        g = g.copy()
        g["company_id"] = company_id
        close = pd.to_numeric(g["close"], errors="coerce")
        y_log = np.log(close.shift(-horizon)) - np.log(close)
        if style == "log":
            y = y_log
            return _winsorize_series(y, winsorize_p) if winsorize_p else y
        return (y_log > 0).astype(int)

    return (
        df.groupby("company_id", sort=False, group_keys=False)
          .apply(lambda g: _one(g, g.name), include_groups=False)
    )

def add_basic_technical_features_grouped(
    df: pd.DataFrame,
    n_lags: int,
    add_indicators: bool
) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()

    def _tech(g: pd.DataFrame, company_id) -> pd.DataFrame:
        g = g.copy()
        g["company_id"] = company_id
        g = g.sort_values("date")
        g["r"] = np.log(g["close"]).diff()

        for i in range(1, n_lags + 1):
            g[f"r_lag_{i}"] = g["r"].shift(i)

        g["r_ma_5"]  = g["r"].rolling(5).mean()
        g["r_ma_10"] = g["r"].rolling(10).mean()
        g["r_ma_21"] = g["r"].rolling(21).mean()
        g["r_std_5"]  = g["r"].rolling(5).std()
        g["r_std_10"] = g["r"].rolling(10).std()
        g["r_std_21"] = g["r"].rolling(21).std()

        g["price_range_10"] = (g["close"].rolling(10).max() / g["close"].rolling(10).min()).apply(np.log)
        g["price_range_21"] = (g["close"].rolling(21).max() / g["close"].rolling(21).min()).apply(np.log)

        if add_indicators:
            delta = g["close"].diff()
            up = delta.clip(lower=0).rolling(14).mean()
            down = (-delta.clip(upper=0)).rolling(14).mean()
            rsi = 100 - (100 / (1 + (up / (down.replace(0, np.nan)))))
            g["rsi_14"] = rsi

            ema12 = g["close"].ewm(span=12, adjust=False).mean()
            ema26 = g["close"].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            hist = macd - signal
            g["macd"] = macd
            g["macd_signal"] = signal
            g["macd_hist"] = hist

        return g

    out = (
        out.groupby("company_id", sort=False, group_keys=False)
           .apply(lambda g: _tech(g, g.name), include_groups=False)
           .reset_index(drop=True)
    )

    tech_cols = [c for c in out.columns if c.startswith(("r_lag_", "r_ma_", "r_std_", "price_range_"))]
    if add_indicators:
        tech_cols += ["rsi_14", "macd", "macd_signal", "macd_hist"]

    return out, tech_cols

# ---------- FUND + MACRO (FUND per company; MACRO global + lag1) ----------
def align_exogenous_grouped(
    df: pd.DataFrame,
    include_changes: bool,
    known_lag_sessions: int = 1,
    change_windows: List[int] = [21, 63],
) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()

    present_fund  = sorted(list(FUND.intersection(out.columns)))
    present_macro = sorted(list(MACRO.intersection(out.columns)))
    created: List[str] = []

    if present_fund:
        def _fund(g: pd.DataFrame, company_id) -> pd.DataFrame:
            g = g.copy()
            g["company_id"] = company_id
            g[present_fund] = g[present_fund].apply(pd.to_numeric, errors="coerce").ffill()
            for col in present_fund:
                g[f"{col}_lag{known_lag_sessions}"] = g[col].shift(known_lag_sessions)
                if include_changes:
                    for w in change_windows:
                        g[f"{col}_chg_{w}d"] = _pct_change_safe(g[col].shift(1), w)
            return g

        out = (
            out.groupby("company_id", sort=False, group_keys=False)
               .apply(lambda g: _fund(g, g.name), include_groups=False)
               .reset_index(drop=True)
        )

        for col in present_fund:
            created.append(f"{col}_lag{known_lag_sessions}")
            if include_changes:
                for w in change_windows:
                    created.append(f"{col}_chg_{w}d")

    if present_macro:
        out[present_macro] = out[present_macro].apply(pd.to_numeric, errors="coerce").ffill()
        for col in present_macro:
            out[f"{col}_lag1"] = out[col].shift(1)
            if include_changes:
                for w in change_windows:
                    out[f"{col}_chg_{w}d"] = _pct_change_safe(out[col].shift(1), w)
        for col in present_macro:
            created.append(f"{col}_lag1")
            if include_changes:
                for w in change_windows:
                    created.append(f"{col}_chg_{w}d")

    return out, created

def add_calendar_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    dow = out["date"].dt.weekday
    out["sin_dow"] = np.sin(2 * np.pi * dow / 5)
    out["cos_dow"] = np.cos(2 * np.pi * dow / 5)

    per = out["date"].dt.to_period("M")
    pos = per.groupby(per).cumcount() + 1
    cnt = per.value_counts().reindex(per).to_numpy()
    mp = pos / cnt
    out["sin_mp"] = np.sin(2 * np.pi * mp)
    out["cos_mp"] = np.cos(2 * np.pi * mp)

    return out, ["sin_dow", "cos_dow", "sin_mp", "cos_mp"]

# ---------- main ----------
def prepare_features(
    df: pd.DataFrame,
    n_lags: int = 30,
    horizon: int = 5,
    target_style: Literal["log", "direction"] = "log",
    winsorize_p: Optional[float] = 0.005,
    drop_ohlcv: bool = False,
    add_indicators: bool = True,
    short_horizon_pack: bool = True,
    remove_constant_features: bool = True,
    known_lag_sessions: int = 1,            # jeśli publikacje są po zamknięciu → daj 2
    add_company_id_feature: bool = False,
    add_sector_id_feature: bool = False
) -> Tuple[pd.DataFrame, List[str], pd.Series, Dict, Optional[str]]:

    df = df.copy()
    # zachowaj id do ewentualnych wag/feature
    keep_meta = [c for c in ("company_id", "sector_id", "date", "close") if c in df.columns]

    df = df.drop(columns=[c for c in ("id", "created_at") if c in df.columns], errors="ignore")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    sort_cols = ["company_id", "date"] if "company_id" in df.columns else ["date"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    if drop_ohlcv:
        df = df.drop(columns=[c for c in OHLCV if c in df.columns], errors="ignore")

    # target per spółka
    y = make_target_grouped(df, horizon=horizon, style=target_style, winsorize_p=winsorize_p if target_style=="log" else None)

    # techniczne per spółka
    df1, tech_cols = add_basic_technical_features_grouped(df, n_lags=n_lags, add_indicators=add_indicators)

    df2, exo_cols = align_exogenous_grouped(
        df1,
        include_changes=not short_horizon_pack,
        known_lag_sessions=known_lag_sessions,
        change_windows=[21, 63]
    )

    df3, cal_cols = add_calendar_features(df2)

    extra_cols = []
    if add_company_id_feature and "company_id" in df3.columns:
        df3["company_id_num"] = pd.to_numeric(df3["company_id"], errors="coerce")
        extra_cols.append("company_id_num")
    if add_sector_id_feature and "sector_id" in df3.columns:
        df3["sector_id_num"] = pd.to_numeric(df3["sector_id"], errors="coerce")
        extra_cols.append("sector_id_num")

    feature_cols = tech_cols + exo_cols + cal_cols + extra_cols

    frame = df3[keep_meta + feature_cols].dropna(subset=feature_cols).copy()
    y = y.loc[frame.index]

    if remove_constant_features:
        frame_no_meta = _drop_constant_cols(frame.drop(columns=keep_meta, errors="ignore"))
        frame = pd.concat([frame[keep_meta], frame_no_meta], axis=1)
        feature_cols = [c for c in feature_cols if c in frame.columns]

    if target_style == "log":
        xgb_params = {"objective": "reg:squarederror", "eval_metric": ["rmse"]}
    else:
        xgb_params = {"objective": "binary:logistic", "eval_metric": ["logloss", "auc"]}

    return frame, feature_cols, y, xgb_params, None
