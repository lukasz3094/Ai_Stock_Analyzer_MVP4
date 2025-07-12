import pandas as pd
import pandas_ta as ta

def transform_market_data(df: pd.DataFrame, existing_keys: set) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.dropna(subset=["close"])

    df["sma_14"] = ta.sma(df["close"], length=14)
    df["ema_14"] = ta.ema(df["close"], length=14)
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    macd = ta.macd(df["close"])
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]

    df["key"] = list(zip(df["company_id"], df["date"]))
    df = df[~df["key"].isin(existing_keys)]
    df = df.drop(columns=["key"])
    df = df.dropna()

    return df
