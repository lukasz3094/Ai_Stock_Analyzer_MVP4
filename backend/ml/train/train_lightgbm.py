import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor, early_stopping

# ---- trading calendar helpers ----
def _session_index_from_table(gpw_calendar: list) -> pd.DatetimeIndex:
    """Convert list of GpwSessions to pd.DatetimeIndex of trading days."""
    if not gpw_calendar:
        return pd.DatetimeIndex([])
    gpw_df = pd.DataFrame([{"trade_date": gs.trade_date, "is_trading_day": gs.is_trading_day} for gs in gpw_calendar])
    gpw_df["trade_date"] = pd.to_datetime(gpw_df["trade_date"]).dt.normalize()
    s = (gpw_df.loc[gpw_df["is_trading_day"].astype(bool)]
               .drop_duplicates("trade_date")
               .sort_values("trade_date"))
    return pd.DatetimeIndex(s["trade_date"])

def _next_trading_days(last_date: pd.Timestamp, n: int, session_idx: pd.DatetimeIndex | None) -> pd.DatetimeIndex:
    """If session_idx is provided (from gpw_sessions), use it; otherwise fallback to business days."""
    ld = pd.to_datetime(last_date).normalize()
    if session_idx is not None and len(session_idx) > 0:
        fut = session_idx[session_idx > ld]
        if len(fut) < n:
            raise ValueError(f"gpw_sessions has only {len(fut)} future sessions; need {n}.")
        return fut[:n]
    return pd.bdate_range(start=ld + pd.Timedelta(days=1), periods=n)

# ---- 2) TRAIN MODEL -----------------------------------------------------------
def train_model(df_feat: pd.DataFrame, feature_cols: list[str]):
    # enforce numeric types
    cols = list(set(["r", "close"] + feature_cols))
    df_feat[cols] = df_feat[cols].apply(pd.to_numeric, errors="coerce").astype("float32")

    # keep DataFrames (names!) to avoid sklearn warning
    X = df_feat[feature_cols]
    y = df_feat["r"].astype("float32")

    split = int(len(df_feat) * 0.8)  # time split
    Xtr, ytr = X.iloc[:split], y.iloc[:split]
    Xte, yte = X.iloc[split:],   y.iloc[split:]

    model = LGBMRegressor(
        n_estimators=4000, learning_rate=0.02,
        max_depth=-1, num_leaves=127,
        colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
        min_data_in_leaf=20, reg_lambda=1.2,
        random_state=42, n_jobs=-1, verbosity=-1
    )

    # early stopping -> best_iteration_ will be set
    model.fit(
        Xtr, ytr,
        eval_set=[(Xte, yte)],
        eval_metric="l1",
        callbacks=[early_stopping(stopping_rounds=300, verbose=False)]
    )

    best_iter = int(getattr(model, "best_iteration_", model.n_estimators_))
    y_hat_te  = model.predict(Xte, num_iteration=best_iter)

    mae  = float(mean_absolute_error(yte, y_hat_te))
    bias = float((y_hat_te - yte).mean())
    q_low, q_high = float(np.percentile(y_hat_te, 1)), float(np.percentile(y_hat_te, 99))

    info = {
        "test_mae_returns": mae,
        "n_train": int(len(ytr)),
        "n_test":  int(len(yte)),
        "bias": bias,
        "q_low": q_low,
        "q_high": q_high,
        "best_iter": best_iter,
    }
    return model, info

# ---- 3) FORECAST NEXT SESSIONS ------------------------------------------------
def forecast_next_sessions(
    model: LGBMRegressor,
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    n_steps: int = 30,
    session_idx: pd.DatetimeIndex | None = None,
    *,
    bias: float = 0.0,
    shrink: float = 0.6,
    clip: tuple[float, float] | None = None
) -> pd.DataFrame:
    # seed window with last n lags of returns
    lag_cols = [c for c in feature_cols if c.startswith("r_lag_")]
    n_lags = max(int(c.split("_")[-1]) for c in lag_cols) if lag_cols else 10

    hist_r = df_feat["r"].to_numpy()
    if len(hist_r) < n_lags:
        raise ValueError("Not enough history to build lagged returns.")
    window = deque(hist_r[-n_lags:].tolist(), maxlen=n_lags)

    last_row = df_feat.iloc[-1].copy()  # contains exogenous + calendar from last session
    last_price = float(df_feat["close"].iloc[-1])
    start_date = df_feat["date"].iloc[-1]
    f_dates = _next_trading_days(start_date, n_steps, session_idx)

    preds = []
    for t, d in enumerate(f_dates):
        # update calendar features
        dow = d.weekday()
        last_row["sin_dow"] = np.sin(2*np.pi*dow/5)
        last_row["cos_dow"] = np.cos(2*np.pi*dow/5)
        # update price-based features from evolving window
        r_arr = np.array(list(window), dtype=float)
        for i in range(1, n_lags + 1):
            last_row[f"r_lag_{i}"] = r_arr[-i]
        last_row["r_ma_5"]  = r_arr[-5:].mean() if len(r_arr) >= 5 else r_arr.mean()
        last_row["r_ma_10"] = r_arr[-10:].mean() if len(r_arr) >= 10 else r_arr.mean()
        last_row["r_std_5"] = r_arr[-5:].std() if len(r_arr) >= 5 else r_arr.std()
        last_row["r_std_10"] = r_arr[-10:].std() if len(r_arr) >= 10 else r_arr.std()

        x = last_row[feature_cols].to_frame().T
        x = x.apply(pd.to_numeric, errors="coerce").astype("float32")

        num_iter = getattr(model, "best_iteration_", None)  # None falls back to full model
        r_pred = float(model.predict(x, num_iteration=num_iter)[0])

        last_price *= np.exp(r_pred)
        preds.append((d, last_price))
        window.append(r_pred)

    return pd.DataFrame({"date": f_dates, "forecast": [p for _, p in preds]})

# ---- 4) PLOT & PRINT ----------------------------------------------------------
def plot_and_print(df_feat: pd.DataFrame, fc_df: pd.DataFrame, company_id: int | str):
    print(f"\nLast session in data: {df_feat['date'].iloc[-1].date()}")
    print("Forecast (price):")
    for dt, p in zip(fc_df["date"].dt.date, fc_df["forecast"]):
        print(f"{dt} - {p:.2f}")

    plt.figure(figsize=(11, 5))
    plt.plot(df_feat["date"].tail(220), df_feat["close"].tail(220), label="History")
    plt.plot(fc_df["date"], fc_df["forecast"], "--", label="Forecast (next sessions)")
    plt.title(f"Company {company_id} — LightGBM on returns")
    plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.tight_layout(); plt.show()

    plt.savefig(f"lightgbm_company_{company_id}_forecast.png")

# ---- EXAMPLE MAIN -------------------------------------------------------------
if __name__ == "__main__":
    from app.branches.prediction_branch.data_loader.features_loader import get_data_for_model_per_company
    from app.services.getters.gpw_calendar_getter import get_gpwsessions_by_start_date
    from ml.train.prepare_features import prepare_features  # your existing function

    company_id = 612
    df_raw = get_data_for_model_per_company(company_id)
    df_raw = df_raw[pd.to_datetime(df_raw["date"]) <= pd.Timestamp("2024-12-31")]  # backtest cutoff

    df_feat, feat_cols = prepare_features(df_raw, n_lags=10)  # (optionally pass wig20, dump_csv_path=...)
    model, info = train_model(df_feat, feat_cols)
    print(f"Test MAE (returns): {info['test_mae_returns']:.6f} | n_train={info['n_train']} n_test={info['n_test']}")
    print(f"Bias={info['bias']:.6e} | clip=({info['q_low']:.4e}, {info['q_high']:.4e}) | best_iter={model.best_iteration_}")

    gpw_calendar = get_gpwsessions_by_start_date(df_raw["date"].min())
    session_idx = _session_index_from_table(gpw_calendar)

    fc_df = forecast_next_sessions(
        model, df_feat, feat_cols, n_steps=30, session_idx=session_idx,
        bias=info["bias"], shrink=0.6, clip=(info["q_low"], info["q_high"])
    )
    plot_and_print(df_feat, fc_df, company_id)
