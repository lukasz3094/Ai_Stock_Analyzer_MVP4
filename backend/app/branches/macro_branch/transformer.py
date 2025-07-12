import pandas as pd

def transform_macro_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    full_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_range)

    df = df.fillna(method="ffill").fillna(method="bfill")
    df = df.reset_index().rename(columns={"index": "date"})
    return df
