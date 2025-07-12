import pandas as pd

def transform_macro_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values("date")
    df = df.fillna(method="ffill").fillna(method="bfill")
    
    return df
