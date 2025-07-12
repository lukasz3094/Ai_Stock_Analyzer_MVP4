import pandas as pd

def transform_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.dropna(subset=["date"])

    company_id = df["company_id"].iloc[0]
    start_date = df["date"].min()
    end_date = df["date"].max()

    full_index = pd.date_range(start=start_date, end=end_date, freq="D")
    df = df.set_index("date").reindex(full_index)
    df["company_id"] = company_id

    df = df.ffill().reset_index().rename(columns={"index": "date"})
    df = df.dropna(subset=["revenue", "operating_profit", "gross_profit", "net_profit", "ebitda"])

    return df
