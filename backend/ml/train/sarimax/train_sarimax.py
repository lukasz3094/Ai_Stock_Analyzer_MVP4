import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import List, Dict, Any, Iterable
from app.services.getters.companies_getter import get_companies_by_group
from app.services.getters.features_getter import get_features_for_company

# Helper Functions
def _row_to_dict(row: Any) -> dict:
    """Best-effort conversion of a DB row object into a dict with column names."""

    if isinstance(row, pd.Series):
        return row.to_dict()

    if hasattr(row, "_mapping"):
        try:
            return dict(row._mapping)  # preserves column names
        except Exception:
            pass

    if hasattr(row, "keys") and hasattr(row, "__iter__"):
        try:
            keys = list(row.keys())
            vals = list(row)
            if len(keys) == len(vals):
                return dict(zip(keys, vals))
        except Exception:
            pass

    if hasattr(row, "_fields"):
        try:
            return {k: getattr(row, k) for k in row._fields}
        except Exception:
            pass

    if hasattr(row, "__dict__"):
        d = {k: v for k, v in vars(row).items() if not k.startswith("_")}
        return d

    if isinstance(row, dict):
        return row

    if isinstance(row, (list, tuple)):
        return {f"col_{i}": v for i, v in enumerate(row)}

    return {"value": row}

def to_dataframe(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()

    if isinstance(obj, dict):
        return pd.DataFrame([obj])

    if isinstance(obj, (list, tuple, set)) or isinstance(obj, Iterable):
        obj_list = list(obj)
        if not obj_list:
            return pd.DataFrame()

        first = obj_list[0]
        if isinstance(first, dict):
            return pd.DataFrame(obj_list)

        normalized = [_row_to_dict(r) for r in obj_list]
        return pd.DataFrame(normalized)

    raise TypeError(f"Unsupported data type from get_features_for_company: {type(obj)}")

def ensure_datetime_index(df: pd.DataFrame, date_col_candidates=('date','Date','trading_date','DATE','TradeDate')):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for c in date_col_candidates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce', utc=False)
            df = df.dropna(subset=[c]).sort_values(c).set_index(c)
            return df

    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors='coerce', utc=False)
            if parsed.notna().sum() > 0:
                df[c] = parsed
                df = df.dropna(subset=[c]).sort_values(c).set_index(c)
                return df
        except Exception:
            continue

    raise KeyError(f"No date column found among: {date_col_candidates} or by auto-detection")

def coerce_numeric_inplace(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')



def prepare_data(
    df_raw: pd.DataFrame,
    target_variable: str,
    exog_variables: List[str],
    forecast_horizon: int
) -> Dict[str, Any]:
    if df_raw.empty:
        return {'status': 'error', 'message': 'Raw data is empty.'}

    if not isinstance(df_raw.index, pd.DatetimeIndex):
        return {'status': 'error', 'message': 'DataFrame index must be DatetimeIndex.'}
    df = df_raw.sort_index()

    available_exog = [c for c in exog_variables if c in df.columns]
    missing_exog = sorted(set(exog_variables) - set(available_exog))
    if missing_exog:
        print(f"[warn] Missing exogenous columns (skipped): {missing_exog}")

    needed = [target_variable] + available_exog
    missing_needed = [c for c in [target_variable] if c not in df.columns]
    if missing_needed:
        return {'status': 'error', 'message': f"Missing target column(s): {missing_needed}"}
    
    for col_to_remove in ['id', 'company_id']:
        if col_to_remove in df.columns:
            df = df.drop(columns=[col_to_remove])

    df_final = df[needed].copy()

    df_final = df_final.ffill().dropna()

    if df_final.shape[0] < (forecast_horizon * 2):
        return {'status': 'error', 'message': f'Insufficient data points ({df_final.shape[0]}) after cleaning.'}

    train_data = df_final.iloc[:-forecast_horizon]
    test_data  = df_final.iloc[-forecast_horizon:]

    y_train = train_data[target_variable]
    X_train = train_data[available_exog] if available_exog else None
    X_test  = test_data[available_exog]  if available_exog else None

    return {
        'status': 'success',
        'y_train': y_train,
        'X_train': X_train,
        'X_test': X_test,
        'last_date': pd.to_datetime(df_final.index[-1]),
        'used_exog': available_exog
    }

def train_and_forecast_sarimax(data: Dict[str, Any], company_ticker: str, forecast_horizon: int) -> pd.DataFrame:
    y_train = data['y_train']
    X_train = data['X_train']  # may be None
    X_test  = data['X_test']   # must be None if X_train is None
    used_exog = data.get('used_exog', [])

    # --- Hyperparameters (tune later / auto_arima) ---
    order = (1, 1, 1)
    seasonal_order = (0, 0, 0, 0)  # s=1 is effectively non-seasonal and avoids s=0 pitfalls

    try:
        model = SARIMAX(
            endog=y_train,
            exog=X_train,  # None is fine
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)

        # If we trained with exog, we must pass exog for forecast
        steps = forecast_horizon
        forecast = results.get_forecast(steps=steps, exog=X_test if used_exog else None)

        # Build tidy frame
        ci = forecast.conf_int(alpha=0.05)
        ci.columns = ['Lower CI (95%)', 'Upper CI (95%)']
        out = pd.DataFrame({
            'Predicted Price': forecast.predicted_mean
        }).join(ci)

        # Metadata & step index
        out['company_ticker'] = company_ticker
        out['forecast_start_date'] = pd.to_datetime(data['last_date']) + pd.Timedelta(days=1)
        out.reset_index(drop=False, inplace=True)
        out.rename(columns={'index': 'forecast_step'}, inplace=True)

        return out

    except Exception as e:
        print(f"ERROR: Failed to train/forecast SARIMAX for {company_ticker}. Error: {e}")
        return pd.DataFrame()

def run_pipeline():
    GROUP_NAME = "wig-banki"
    TARGET_VAR = 'close'
    EXOG_VARS = ['gdp', 'cpi', 'unemployment_rate', 'interest_rate', 'exchange_rate_eur']
    FORECAST_HORIZON = 14

    companies = get_companies_by_group(GROUP_NAME)
    print(f"Found {len(companies)} companies to process.")

    all_forecasts = []

    for company in companies:
        print(f"\n--- Processing {company.ticker} ({getattr(company, 'name', company.ticker)}) ---")

        # 1) Normalize to DataFrame safely
        raw = get_features_for_company(company.id)
        
        try:
            df = to_dataframe(raw)
        except Exception as e:
            print(f"Skipping {company.ticker}: cannot convert data to DataFrame: {e}")
            continue

        if df.empty:
            print(f"Skipping {company.ticker}: empty dataset.")
            continue

        # 2) Ensure datetime index
        try:
            df = ensure_datetime_index(df, ('date','Date','trading_date'))
        except Exception as e:
            print(f"Skipping {company.ticker}: {e}")
            continue

        # 3) Coerce numeric of target + exog if present
        numeric_cols = [TARGET_VAR] + EXOG_VARS
        coerce_numeric_inplace(df, numeric_cols)

        # 4) Prepare data
        data_prep_result = prepare_data(df, TARGET_VAR, EXOG_VARS, FORECAST_HORIZON)
        if data_prep_result['status'] == 'error':
            print(f"Skipping {company.ticker}: {data_prep_result['message']}")
            continue

        # 5) Train & forecast
        forecast_df = train_and_forecast_sarimax(data_prep_result, company.ticker, FORECAST_HORIZON)
        if not forecast_df.empty:
            all_forecasts.append(forecast_df)
            print(f"Successfully forecasted for {company.ticker}.")

    # 6) Save results
    if all_forecasts:
        final_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
        output_filename = 'wig_banki_sarimax_14day_forecasts.csv'
        final_forecasts_df.to_csv(output_filename, index=False)
        print(f"\n✅ Pipeline complete. Results saved to {output_filename}")
        print("\nFinal Forecasts Head:")
        try:
            print(final_forecasts_df.head().to_markdown(numalign="left", stralign="left"))
        except Exception:
            print(final_forecasts_df.head())
    else:
        print("\n❌ Pipeline finished, but no successful forecasts were generated.")


if __name__ == "__main__":
    run_pipeline()