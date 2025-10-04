import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm

import warnings
warnings.filterwarnings(
    "ignore",
    message="Non-stationary starting autoregressive parameters",
    category=UserWarning
)

# ---------- tqdm callback for SARIMAX ----------
def sarimax_tqdm(maxiter: int, desc: str = "SARIMAX"):
    pbar = tqdm(total=maxiter, desc=desc, unit="iter")
    def _cb(params):  # statsmodels calls this each iteration
        pbar.update(1)
    _cb.close = pbar.close
    return _cb

# ---------- trading calendar ----------
def _session_index_from_table(gpw_calendar: list) -> pd.DatetimeIndex:
    if not gpw_calendar:
        return pd.DatetimeIndex([])
    gpw_df = pd.DataFrame([{"trade_date": gs.trade_date, "is_trading_day": gs.is_trading_day} for gs in gpw_calendar])
    gpw_df["trade_date"] = pd.to_datetime(gpw_df["trade_date"]).dt.normalize()
    s = (gpw_df.loc[gpw_df["is_trading_day"].astype(bool)]
               .drop_duplicates("trade_date")
               .sort_values("trade_date"))
    return pd.DatetimeIndex(s["trade_date"])

def _next_trading_days(last_date: pd.Timestamp, n: int, session_idx: pd.DatetimeIndex | None) -> pd.DatetimeIndex:
    ld = pd.to_datetime(last_date).normalize()
    if session_idx is not None and len(session_idx) > 0:
        fut = session_idx[session_idx > ld]
        if len(fut) < n: raise ValueError(f"gpw_sessions has only {len(fut)} future sessions; need {n}.")
        return fut[:n]
    return pd.bdate_range(start=ld + pd.Timedelta(days=1), periods=n)

def _calendar_feats_for_dates(dts: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=dts)
    dow = dts.weekday
    out["sin_dow"] = np.sin(2*np.pi*dow/5)
    out["cos_dow"] = np.cos(2*np.pi*dow/5)
    # business-day position within month (approx)
    vals = []
    for d in dts:
        month = pd.bdate_range(d.replace(day=1), d.replace(day=28) + pd.offsets.MonthEnd(1))
        pos = np.searchsorted(month.values, d.to_datetime64()) + 1
        cnt = max(1, month.size)
        vals.append(pos / cnt)
    mp = np.array(vals)
    out["sin_mp"] = np.sin(2*np.pi*mp)
    out["cos_mp"] = np.cos(2*np.pi*mp)
    return out

# ---------- exogenous selection & scaling ----------
def _pick_arima_exog_cols(feature_cols):
    keep = []
    for c in feature_cols:
        if c in ("sin_dow","cos_dow","sin_mp","cos_mp"):
            keep.append(c)
        elif c.endswith("_lag1") or c.endswith("_chg_21d") or c.endswith("_chg_63d"):
            keep.append(c)
    return keep

def _zscore_fit(X: pd.DataFrame) -> dict:
    mu = X.mean(axis=0)
    sd = X.std(axis=0).replace(0, 1.0)
    return {"mu": mu, "sd": sd}

def _zscore_apply(X: pd.DataFrame, stats: dict) -> pd.DataFrame:
    X2 = X.copy()
    # add missing cols with zeros (after standardization)
    for c in stats["mu"].index:
        if c not in X2.columns:
            X2[c] = stats["mu"][c]
    X2 = X2[stats["mu"].index]  # order align
    Z = (X2 - stats["mu"]) / stats["sd"]
    return Z.astype(float)

# ---------- 1) TRAIN ----------
def train_model(df_feat: pd.DataFrame, feature_cols: list[str], candidate_orders=((1,0,0),(0,0,1),(1,0,1))):
    exog_cols = _pick_arima_exog_cols(feature_cols)

    # numeric & clean
    need = set(["date","close","r"] + exog_cols)
    df = df_feat[[c for c in df_feat.columns if c in need]].copy()
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["r"]).reset_index(drop=True)

    # split
    split = int(len(df)*0.8)
    
    y = df["r"].clip(lower=df["r"].quantile(0.01), upper=df["r"].quantile(0.99))
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    # y_tr, y_te = df["r"].iloc[:split], df["r"].iloc[split:]

    # standardize y (helps optimizer) and exog
    y_mu, y_sd = y_tr.mean(), y_tr.std() or 1.0
    y_tr_z = (y_tr - y_mu) / y_sd
    y_te_z = (y_te - y_mu) / y_sd

    X_tr = df[exog_cols].iloc[:split] if exog_cols else None
    X_te = df[exog_cols].iloc[split:] if exog_cols else None

    xstats = None
    if X_tr is not None:
        X_tr = X_tr.ffill().bfill()
        X_te = X_te.ffill().bfill()
        xstats = _zscore_fit(X_tr)
        X_tr_z = _zscore_apply(X_tr, xstats)
        X_te_z = _zscore_apply(X_te, xstats)
    else:
        X_tr_z = X_te_z = None

    best = {"aic": np.inf, "res": None, "order": None}
    for order in candidate_orders:
        try:
            mod = SARIMAX(
                endog=y_tr_z, exog=X_tr_z,
                order=order,
                trend="n",  # zero-mean after zscore
                enforce_stationarity=True,
                enforce_invertibility=True
            )

            # mod = SARIMAX(
            #     endog=y_tr_z, exog=X_tr_z, order=order,
            #     seasonal_order=(1,0,1,5),  # <- add this line to try weekly seasonality
            #     trend="n", enforce_stationarity=True, enforce_invertibility=True
            # )

            cb = sarimax_tqdm(maxiter=500, desc=f"SARIMAX{order}")
            res = mod.fit(disp=False, maxiter=500, callback=cb)
            cb.close()

            if np.isfinite(res.aic) and res.aic < best["aic"]:
                best = {"aic": res.aic, "res": res, "order": order}
        except Exception:
            continue

    if best["res"] is None:
        # fallback: no exog, simplest AR(1)
        cb = sarimax_tqdm(maxiter=500, desc="SARIMAX(1,0,0) fallback")
        mod = SARIMAX(y_tr_z, order=(1,0,0), trend="n",
                      enforce_stationarity=True, enforce_invertibility=True)
        res = mod.fit(method="lbfgs", maxiter=500, disp=True, callback=cb)
        cb.close()
        
        best = {"aic": res.aic, "res": res, "order": (1,0,0)}
        X_te_z = None  # since we refit without exog

    # 1-step-ahead predictions on test block
    fc = best["res"].get_forecast(steps=len(y_te_z), exog=X_te_z)
    yhat_z = fc.predicted_mean.to_numpy()
    yhat = yhat_z * y_sd + y_mu

    mae  = float(mean_absolute_error(y_te.to_numpy(), yhat))
    bias = float((yhat - y_te.to_numpy()).mean())
    q_low, q_high = float(np.percentile(yhat, 1)), float(np.percentile(yhat, 99))

    info = {
        "test_mae_returns": mae,
        "n_train": int(len(y_tr)),
        "n_test":  int(len(y_te)),
        "bias": bias,
        "q_low": q_low,
        "q_high": q_high,
        "order": best["order"],
        "exog_cols": exog_cols,
        "y_mu": float(y_mu),
        "y_sd": float(y_sd),
        "xstats": xstats,
    }
    return best["res"], info

# ---------- 2) FORECAST ----------
def forecast_next_sessions(
    res, df_feat: pd.DataFrame, feature_cols: list[str],
    n_steps: int = 30, session_idx: pd.DatetimeIndex | None = None,
    *, bias: float = 0.0, shrink: float = 1.0, clip: tuple[float, float] | None = None,
    y_mu: float = 0.0, y_sd: float = 1.0, xstats: dict | None = None
) -> pd.DataFrame:
    exog_cols = _pick_arima_exog_cols(feature_cols)

    last_price = float(df_feat["close"].iloc[-1])
    start_date = pd.to_datetime(df_feat["date"].iloc[-1]).normalize()
    f_dates = _next_trading_days(start_date, n_steps, session_idx)

    # future exog (frozen last known *_lag1 + fresh calendar)
    if exog_cols:
        last = df_feat.iloc[-1]
        future = _calendar_feats_for_dates(f_dates)
        for c in exog_cols:
            if c not in future.columns:
                future[c] = pd.to_numeric(last.get(c, np.nan), errors="coerce")
        future = future[exog_cols].ffill().bfill()
        X_fut = _zscore_apply(future, xstats) if xstats is not None else future
    else:
        X_fut = None

    fc = res.get_forecast(steps=n_steps, exog=X_fut)
    r_pred_z = fc.predicted_mean.to_numpy().astype(float)
    r_pred = r_pred_z * y_sd + y_mu
    r_pred = (r_pred - bias) * shrink
    if clip is not None:
        r_pred = np.clip(r_pred, clip[0], clip[1])

    prices, p = [], last_price
    for r in r_pred:
        p *= float(np.exp(r))
        prices.append(p)
    return pd.DataFrame({"date": f_dates, "forecast": prices})

# ---------- 3) PLOT & PRINT ----------
def plot_and_print(df_feat: pd.DataFrame, fc_df: pd.DataFrame, company_id: int | str):
    print(f"\nLast session in data: {pd.to_datetime(df_feat['date'].iloc[-1]).date()}")
    print("Forecast (price):")
    for dt, p in zip(fc_df["date"].dt.date, fc_df["forecast"]):
        print(f"{dt} - {p:.2f}")
    plt.figure(figsize=(11,5))
    plt.plot(pd.to_datetime(df_feat["date"]).tail(220), df_feat["close"].tail(220), label="History")
    plt.plot(fc_df["date"], fc_df["forecast"], "--", label="Forecast (next sessions)")
    plt.title(f"Company {company_id} — SARIMAX (stabilized)")
    plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.tight_layout(); plt.show()

# ---------- EXAMPLE MAIN ----------
if __name__ == "__main__":
    from app.branches.prediction_branch.data_loader.features_loader import get_data_for_model_per_company
    from app.services.getters.gpw_calendar_getter import get_gpwsessions_by_start_date
    from ml.train.prepare_features import prepare_features

    company_id = 612
    df_raw = get_data_for_model_per_company(company_id)
    df_raw = df_raw[pd.to_datetime(df_raw["date"]) <= pd.Timestamp("2024-12-31")]

    # prepare_features() should create: r, calendar sin/cos, and *_lag1 for macro/fund (optional *_chg_*).
    df_feat, feat_cols = prepare_features(df_raw, n_lags=10)

    res, info = train_model(
        df_feat, feat_cols,
        candidate_orders=((1,0,0),(0,0,1),(1,0,1),(2,0,1),(1,0,2))
    )

    print(f"Test MAE (returns): {info['test_mae_returns']:.6f} | n_train={info['n_train']} n_test={info['n_test']}")
    print(f"Bias={info['bias']:.6e} | clip=({info['q_low']:.4e}, {info['q_high']:.4e}) | order={info['order']}")

    gpw_calendar = get_gpwsessions_by_start_date(df_raw["date"].min())
    session_idx = _session_index_from_table(gpw_calendar)

    fc_df = forecast_next_sessions(
        res, df_feat, feat_cols, n_steps=30, session_idx=session_idx,
        bias=info["bias"], shrink=1.0, clip=None,
        y_mu=info["y_mu"], y_sd=info["y_sd"], xstats=info["xstats"]
    )
    plot_and_print(df_feat, fc_df, company_id)
