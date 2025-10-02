import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- GPW calendar (no external deps) ---------------- #
def _easter_sunday(year: int) -> pd.Timestamp:
    a = year % 19; b = year // 100; c = year % 100
    d = b // 4; e = b % 4; f = (b + 8) // 25; g = (b - f + 1) // 3
    h = (19*a + b - d - g + 15) % 30; i = c // 4; k = c % 4
    l = (32 + 2*e + 2*i - h - k) % 7; m = (a + 11*h + 22*l) // 451
    month = (h + l - 7*m + 114) // 31; day = 1 + ((h + l - 7*m + 114) % 31)
    return pd.Timestamp(year=int(year), month=int(month), day=int(day))

def _polish_holidays(year: int) -> set:
    easter = _easter_sunday(year)
    easter_mon = easter + pd.Timedelta(days=1)
    corpus_christi = easter + pd.Timedelta(days=60)
    fixed = {
        pd.Timestamp(year, 1, 1), pd.Timestamp(year, 1, 6),
        pd.Timestamp(year, 5, 1), pd.Timestamp(year, 5, 3),
        pd.Timestamp(year, 8, 15), pd.Timestamp(year, 11, 1),
        pd.Timestamp(year, 11, 11), pd.Timestamp(year, 12, 25),
        pd.Timestamp(year, 12, 26),
    }
    moving = {easter_mon.normalize(), corpus_christi.normalize()}
    return {d.normalize() for d in fixed | moving}

class GPWCalendar:
    def __init__(self, extra_closed=None):
        self.extra_closed = {pd.Timestamp(d).normalize() for d in (extra_closed or [])}
    def _year_range(self, start: pd.Timestamp, end: pd.Timestamp):
        return range(start.year, end.year + 1)
    def valid_days(self, start_date, end_date):
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        bdays = pd.bdate_range(start=start, end=end)
        hols = set()
        for y in self._year_range(start, end):
            hols |= _polish_holidays(y)
        hols |= self.extra_closed
        mask = ~bdays.normalize().isin(sorted(hols))
        return pd.DatetimeIndex(bdays[mask])

GPW_CALENDAR = GPWCalendar(extra_closed=[])

def next_trading_days(start_date, n=30):
    try:
        s = pd.Timestamp(start_date).normalize() + pd.Timedelta(days=1)
        sched = GPW_CALENDAR.valid_days(start_date=s, end_date=s + pd.Timedelta(days=120))
        return pd.DatetimeIndex(sched[:n]).tz_localize(None)
    except Exception:
        return pd.bdate_range(start=pd.Timestamp(start_date) + pd.Timedelta(days=1), periods=n)

# ---------------- features: price + fundamentals + macro ---------------- #

# Toggle this to remove OHLCV predictors from the feature set:
DROP_OHLCV = False   # <-- comment out to KEEP open/high/low/volume in features

FUND_COLS_CANDIDATES = {
    "revenue", "operating_profit", "gross_profit", "net_profit", "ebitda"
}
MACRO_COLS_CANDIDATES = {
    "gdp", "cpi", "unemployment_rate", "interest_rate",
    "exchange_rate_eur", "exchange_rate_usd"
}
OHLCV_COLS = {"open", "high", "low", "volume"}

def _pct_change_safe(s, periods):
    out = s.pct_change(periods=periods)
    return out.replace([np.inf, -np.inf], np.nan)

def make_features_with_exog(df, n_lags=10, ma_windows=(5,10)):
    """
    Builds:
      - return lags + rolling mean/std of returns
      - fundamentals/macro (shifted by 1 day) + 21/63d pct-change where applicable
    """

    print(df.columns.tolist())

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)

    # target base
    df["r"] = np.log(df["close"]).diff()

    # price-based AR features
    for i in range(1, n_lags + 1):
        df[f"r_lag_{i}"] = df["r"].shift(i)
    for w in ma_windows:
        df[f"r_ma_{w}"] = df["r"].rolling(w).mean()
        df[f"r_std_{w}"] = df["r"].rolling(w).std()

    # Identify available exogenous cols in this company's frame
    present_fund_cols = sorted(list(FUND_COLS_CANDIDATES.intersection(df.columns)))
    present_macro_cols = sorted(list(MACRO_COLS_CANDIDATES.intersection(df.columns)))

    # shift(1) to avoid look-ahead (use yesterday's info)
    exog_cols_created = []
    for col in present_fund_cols + present_macro_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        df[f"{col}_lag1"] = s.shift(1)
        exog_cols_created.append(f"{col}_lag1")
        # 21d and 63d pct-change (also based on shifted series)
        df[f"{col}_chg_21d"] = _pct_change_safe(s.shift(1), 21)
        df[f"{col}_chg_63d"] = _pct_change_safe(s.shift(1), 63)
        exog_cols_created += [f"{col}_chg_21d", f"{col}_chg_63d"]

    # optionally drop OHLCV predictors (they'll still be used to compute returns)
    if DROP_OHLCV:
        df = df.drop(columns=[c for c in OHLCV_COLS if c in df.columns], errors="ignore")

    # Build final feature list
    price_feats = [c for c in df.columns if c.startswith(("r_lag_","r_ma_","r_std_"))]
    exog_feats = [c for c in exog_cols_created if c in df.columns]
    feature_cols = price_feats + exog_feats

    # clean NA rows after all shifts/rolls
    df = df.dropna(subset=["r"] + feature_cols).reset_index(drop=True)
    return df, feature_cols

# ---------------- training & 30d recursive forecast (XGBoost) ---------------- #

def train_and_forecast_30d_xgb(df, n_lags=10, company_id=None):
    """
    df is what your loader returns (already filtered by Boruta/ALWAYS_KEEP).
    This function WILL use fundamentals/macro if present in df.
    """
    if df.empty:
        raise ValueError("Empty DataFrame received.")

    # Keep everything (do NOT slice to ['date','close'] anymore)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)

    # build features
    df_feat, feature_cols = make_features_with_exog(df, n_lags=n_lags)

    # split
    X = df_feat[feature_cols].to_numpy()
    y = df_feat["r"].to_numpy()
    split = int(len(df_feat)*0.8)
    Xtr, ytr = X[:split], y[:split]
    Xte, yte = X[split:], y[split:]

    # model
    model = xgb.XGBRegressor(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(Xtr, ytr)
    mae = mean_absolute_error(yte, model.predict(Xte))
    print(f"XGBoost MAE (returns) {'for company ' + str(company_id) if company_id else ''}: {mae:.6f}")

    # prepare recursive pieces
    hist_r = df_feat["r"].values
    if len(hist_r) < n_lags:
        raise ValueError("Not enough history to build lagged returns.")
    window = deque(hist_r[-n_lags:].tolist(), maxlen=n_lags)

    last_price = float(df_feat["close"].iloc[-1])
    last_row = df_feat.iloc[-1].copy()   # contains the exogenous features in the right shape

    # Helper to construct next feature vector:
    #   - update r_lag_*, r_ma_*, r_std_* from the evolving window
    #   - keep exogenous (funds/macro) frozen at last known values
    def build_feat():
        arr = [window[-i] for i in range(1, n_lags+1)]
        r = np.array(window, dtype=float)
        ma5  = r[-5:].mean() if len(r)>=5 else r.mean()
        ma10 = r[-10:].mean() if len(r)>=10 else r.mean()
        std5 = r[-5:].std() if len(r)>=5 else r.std()
        std10= r[-10:].std() if len(r)>=10 else r.std()

        temp = last_row.copy()
        for i in range(1, n_lags+1):
            temp[f"r_lag_{i}"] = arr[i-1]
        temp["r_ma_5"]  = ma5
        temp["r_ma_10"] = ma10
        temp["r_std_5"] = std5
        temp["r_std_10"]= std10
        return temp[feature_cols].to_numpy(dtype=float).reshape(1, -1)

    f_dates = next_trading_days(df_feat["date"].iloc[-1], n=30)

    preds_price = []
    for _ in f_dates:
        x = build_feat()
        r_pred = float(model.predict(x)[0])
        last_price *= np.exp(r_pred)
        preds_price.append(last_price)
        window.append(r_pred)

    print(f"\nLast date in data: {df_feat['date'].iloc[-1].date()}")
    print("30-day forecast (XGBoost, price):")
    for dt, p in zip(f_dates.date, preds_price):
        print(f"{dt} - {p:.2f}")

    plt.figure(figsize=(11,5))
    plt.plot(df_feat["date"].tail(220), df_feat["close"].tail(220), label="History")
    plt.plot(f_dates, preds_price, "--", label="Forecast (30d, XGBoost + funda/macro)")
    title = f"Company {company_id} — 30d Forecast (XGBoost on returns + exog)"
    plt.title(title); plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.tight_layout(); plt.show()

    # quick peek at most important features to confirm usage
    try:
        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(20)
        print("\nTop feature importances:")
        print(imp)
    except Exception:
        pass

# ---------------- run ---------------- #
if __name__ == "__main__":
    from app.branches.prediction_branch.data_loader.features_loader import get_data_for_model_per_company
    company_id = 612
    df = get_data_for_model_per_company(company_id)
    df = df[pd.to_datetime(df["date"]) <= pd.Timestamp("2024-12-31")]
    train_and_forecast_30d_xgb(df, n_lags=10, company_id=company_id)
