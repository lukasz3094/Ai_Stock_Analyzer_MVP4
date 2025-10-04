# train_catboost.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

# --- GPW calendar helpers ---
def _session_index_from_table(gpw_calendar: list) -> pd.DatetimeIndex:
    if not gpw_calendar:
        return pd.DatetimeIndex([])
    df = pd.DataFrame([{"trade_date": g.trade_date, "is_trading_day": g.is_trading_day} for g in gpw_calendar])
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    df = df.loc[df["is_trading_day"].astype(bool)].drop_duplicates("trade_date").sort_values("trade_date")
    return pd.DatetimeIndex(df["trade_date"])

def _next_trading_days(last_date: pd.Timestamp, n: int, session_idx: pd.DatetimeIndex | None) -> pd.DatetimeIndex:
    ld = pd.to_datetime(last_date).normalize()
    if session_idx is not None and len(session_idx) > 0:
        fut = session_idx[session_idx > ld]
        if len(fut) < n:
            raise ValueError(f"gpw_sessions has only {len(fut)} future sessions; need {n}.")
        return fut[:n]
    return pd.bdate_range(start=ld + pd.Timedelta(days=1), periods=n)

def _month_phase(d: pd.Timestamp, session_idx: pd.DatetimeIndex | None) -> float:
    """Return position in trading month in [0,1]. Uses GPW sessions if provided, otherwise bdays."""
    if session_idx is not None and len(session_idx) > 0:
        mask = (session_idx.year == d.year) & (session_idx.month == d.month)
        month_days = session_idx[mask]
        if len(month_days) == 0:
            # fallback to business days that month
            month_days = pd.bdate_range(d.to_period("M").start_time, d.to_period("M").end_time)
        pos = int(np.searchsorted(month_days.values, d.to_datetime64())) + 1
        cnt = len(month_days)
    else:
        month_days = pd.bdate_range(d.to_period("M").start_time, d.to_period("M").end_time)
        pos = int((month_days <= d).sum())
        cnt = len(month_days)
    return pos / max(cnt, 1)

# --- 1) Train ---
def train_model(df_feat: pd.DataFrame, feature_cols: list[str]):
    X = df_feat[feature_cols].apply(pd.to_numeric, errors="coerce").astype("float32")
    y = df_feat["r"].apply(pd.to_numeric, errors="coerce").astype("float32")
    split = int(len(df_feat) * 0.8)
    Xtr, ytr = X.iloc[:split], y.iloc[:split]
    Xte, yte = X.iloc[split:], y.iloc[split:]

    model = CatBoostRegressor(
        iterations=4000,
        learning_rate=0.03,
        depth=6,
        loss_function="MAE",
        eval_metric="MAE",
        od_type="Iter", od_wait=300,  # early stopping
        random_seed=42,
        verbose=200,                  # progress log every 200 iters
        thread_count=-1
    )
    model.fit(Xtr, ytr, eval_set=(Xte, yte), use_best_model=True)
    y_hat = model.predict(Xte)
    info = {
        "test_mae_returns": float(mean_absolute_error(yte, y_hat)),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "best_iter": int(getattr(model, "best_iteration_", model.tree_count_)),
    }
    return model, info

# --- 2) Forecast next sessions (recursive on returns) ---
def forecast_next_sessions(
    model: CatBoostRegressor,
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    n_steps: int = 30,
    session_idx: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    lag_cols = [c for c in feature_cols if c.startswith("r_lag_")]
    n_lags = max(int(c.split("_")[-1]) for c in lag_cols) if lag_cols else 10

    hist_r = df_feat["r"].to_numpy()
    if len(hist_r) < n_lags:
        raise ValueError("Not enough history to build lagged returns.")
    window = deque(hist_r[-n_lags:].tolist(), maxlen=n_lags)

    last_row   = df_feat.iloc[-1].copy()
    last_price = float(df_feat["close"].iloc[-1])
    start_date = pd.to_datetime(df_feat["date"].iloc[-1])
    f_dates    = _next_trading_days(start_date, n_steps, session_idx)

    preds = []
    for d in f_dates:
        # update calendar features
        dow = d.weekday()
        last_row["sin_dow"] = np.sin(2*np.pi * dow/5)
        last_row["cos_dow"] = np.cos(2*np.pi * dow/5)
        mp = _month_phase(d, session_idx)
        last_row["sin_mp"] = np.sin(2*np.pi * mp)
        last_row["cos_mp"] = np.cos(2*np.pi * mp)

        # update price-based features from evolving window
        r_arr = np.array(list(window), dtype=float)
        for i in range(1, n_lags + 1):
            last_row[f"r_lag_{i}"] = r_arr[-i]
        last_row["r_ma_5"]  = r_arr[-5:].mean() if len(r_arr) >= 5 else r_arr.mean()
        last_row["r_ma_10"] = r_arr[-10:].mean() if len(r_arr) >= 10 else r_arr.mean()
        last_row["r_std_5"] = r_arr[-5:].std() if len(r_arr) >= 5 else r_arr.std()
        last_row["r_std_10"] = r_arr[-10:].std() if len(r_arr) >= 10 else r_arr.std()

        x_df = last_row[feature_cols].to_frame().T.apply(pd.to_numeric, errors="coerce").astype("float32")
        r_pred = float(model.predict(x_df)[0])

        last_price *= np.exp(r_pred)
        preds.append((d, last_price))
        window.append(r_pred)

    return pd.DataFrame({"date": f_dates, "forecast": [p for _, p in preds]})

# --- 3) Plot & print ---
def plot_and_print(df_feat: pd.DataFrame, fc_df: pd.DataFrame, company_id: int | str):
    print(f"\nLast session in data: {pd.to_datetime(df_feat['date'].iloc[-1]).date()}")
    print("Forecast (price):")
    for dt, p in zip(fc_df["date"].dt.date, fc_df["forecast"]):
        print(f"{dt} - {p:.2f}")

    plt.figure(figsize=(11, 5))
    plt.plot(pd.to_datetime(df_feat["date"]).tail(220), df_feat["close"].tail(220), label="History")
    plt.plot(fc_df["date"], fc_df["forecast"], "--", label="Forecast (next sessions)")
    plt.title(f"Company {company_id} — CatBoost on returns")
    plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.tight_layout(); plt.show()

# --- Example main ---
if __name__ == "__main__":
    from app.branches.prediction_branch.data_loader.features_loader import get_data_for_model_per_company
    from app.services.getters.gpw_calendar_getter import get_gpwsessions_by_start_date
    # use your global prepare_features
    from ml.train.prepare_features import prepare_features

    company_id = 612
    df_raw = get_data_for_model_per_company(company_id)
    df_raw = df_raw[pd.to_datetime(df_raw["date"]) <= pd.Timestamp("2024-12-31")]

    df_feat, feat_cols = prepare_features(df_raw, n_lags=10)
    model, info = train_model(df_feat, feat_cols)
    print(f"Test MAE (returns): {info['test_mae_returns']:.6f} | n_train={info['n_train']} n_test={info['n_test']} | best_iter={info['best_iter']}")

    gpw_calendar = get_gpwsessions_by_start_date(df_raw["date"].min())
    session_idx  = _session_index_from_table(gpw_calendar)

    fc_df = forecast_next_sessions(model, df_feat, feat_cols, n_steps=30, session_idx=session_idx)
    plot_and_print(df_feat, fc_df, company_id)
