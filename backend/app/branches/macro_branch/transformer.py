import pandas as pd

def transform_fundamentals(df: pd.DataFrame) -> pd.DataFrame:



    
    

    cols_to_replace = ["revenue", "operating_profit", "gross_profit", "net_profit", "ebitda"]
    for col in cols_to_replace:
        if col in df.columns:
            df[col] = df[col].replace(0, pd.NA)

    df = df.ffill().reset_index().rename(columns={"index": "date"})
    df = df.dropna(subset=cols_to_replace)

    return df

def transform_macro_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.dropna(subset=["date"])

    start_date = df["date"].min()
    end_date = df["date"].max()

    full_index = pd.date_range(start=start_date, end=end_date, freq="D")
    df = df.set_index("date").reindex(full_index)

    cols_to_replace = ["gdp", "cpi", "unemployment_rate", "interest_rate", "exchange_rate_eur", "exchange_rate_usd"]
    for col in cols_to_replace:
        if col in df.columns:
            df[col] = df[col].replace(0, pd.NA)

    df = df.ffill().reset_index().rename(columns={"index": "date"})
    df = df.dropna(subset=cols_to_replace)

    return df
