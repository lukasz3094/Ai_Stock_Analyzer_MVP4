from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from app.services.getters.companies_getter import get_companies_by_group
from app.branches.fundamentals_branch.extractor import load_fundamentals
from app.branches.macro_branch.extractor import load_macro_data
from app.branches.market_branch.extractor import load_market_data
from app.db.models import FeaturesFinalPrepared
from app.core.config import SessionLocal
from app.core.logger import logger


def _to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col]).dt.normalize()
    return df

def _prep_market(df: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: company_id, date, open, high, low, close, volume
    if df.empty:
        return df
    df = df.copy()
    df = _to_datetime(df, "date").sort_values("date")
    # Ensure numeric types
    for c in ["close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop exact duplicates per date
    df = df.drop_duplicates(subset=["company_id", "date"], keep="last")
    return df

def _prep_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: company_id, date, fundamentals...
    if df.empty:
        return df
    df = df.copy()
    df = _to_datetime(df, "date").sort_values("date")
    # Quarterly: keep only last record if multiple same-date rows
    df = df.drop_duplicates(subset=["company_id", "date"], keep="last")
    # cast numeric cols
    for c in df.columns:
        if c not in ("company_id", "date"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _prep_macro(df: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: date, gdp, cpi, unemployment_rate, interest_rate, exchange_rate_eur, exchange_rate_usd
    if df.empty:
        return df
    df = df.copy()
    df = _to_datetime(df, "date").sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _merge_asof_on_market_dates(
    market: pd.DataFrame,
    fundamentals: pd.DataFrame,
    macro: pd.DataFrame
) -> pd.DataFrame:
    """
    Build backbone = all market trading dates for this company.
    Join fundamentals & macro using merge_asof(direction='backward') to get last-known values.
    """
    # Safety: require essential columns
    if market.empty:
        return pd.DataFrame()

    # Backbone
    backbone = market[["company_id", "date", "close"]].copy()

    # ---- Fundamentals by company_id with ASOF (backward)
    # merge_asof supports grouping via 'by' -> per company_id
    if not fundamentals.empty:
        # must be sorted by merge key
        fundamentals = fundamentals.sort_values(["company_id", "date"])
        backbone = pd.merge_asof(
            backbone.sort_values(["company_id", "date"]),
            fundamentals.sort_values(["company_id", "date"]),
            on="date",
            by="company_id",
            direction="backward",
            allow_exact_matches=True
        )
    else:
        # Ensure columns exist even if empty (later insert needs them)
        for c in ["interest_income","fee_income","banking_result","gross_profit","net_profit",
                  "amortization","assets","equity","shares_thousands","bvps","eps"]:
            if c not in backbone.columns:
                backbone[c] = np.nan

    # ---- Macro with ASOF (backward) — macro is economy-wide (no company_id)
    if not macro.empty:
        backbone = pd.merge_asof(
            backbone.sort_values(["date"]),
            macro.sort_values(["date"]),
            on="date",
            direction="backward",
            allow_exact_matches=True
        )
    else:
        for c in ["gdp","cpi","unemployment_rate","interest_rate","exchange_rate_eur","exchange_rate_usd"]:
            if c not in backbone.columns:
                backbone[c] = np.nan

    # Gentle ffill for any NaNs left because we did backward only; do NOT forward-fill market OHLCV
    fund_cols = ["interest_income","fee_income","banking_result","gross_profit","net_profit",
                 "amortization","assets","equity","shares_thousands","bvps","eps"]
    macro_cols = ["gdp","cpi","unemployment_rate","interest_rate","exchange_rate_eur","exchange_rate_usd"]

    # forward-fill fundamentals/macro within this company's time frame (no leakage—still backward-known values)
    backbone = backbone.sort_values("date")
    backbone[fund_cols] = backbone[fund_cols].ffill()
    backbone[macro_cols] = backbone[macro_cols].ffill()

    # Optionally, drop rows without a close price (non-trading dates shouldn't exist in market backbone, but just in case)
    backbone = backbone[~backbone["close"].isna()]

    # Final dedup (safety)
    backbone = backbone.drop_duplicates(subset=["company_id", "date"], keep="last")

    return backbone.sort_values("date")


def _nan_to_none_record(d: dict) -> dict:
    # Convert numpy types and NaNs to native types acceptable by SQLAlchemy/Postgres
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.float32, np.float64)):
            v = float(v)
        elif isinstance(v, (np.integer, np.int32, np.int64)):
            v = int(v)
        if isinstance(v, pd.Timestamp):
            v = v.to_pydatetime().date()
        if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) or pd.isna(v):
            v = None
        out[k] = v
    return out


class FundamentalsBranchPipeline:
    def __init__(self, batch_size: int = 1000):
        self.db = SessionLocal()
        self.batch_size = batch_size

    def run(self):
        try:
            logger.info("Pobieranie spółek z grupy WIG banki...")
            companies = get_companies_by_group("wig-banki")

            logger.info("Pobieranie danych makroekonomicznych...")
            macro_data = _prep_macro(load_macro_data())

            for company in companies:
                company_id = company.id
                ticker = getattr(company, "ticker", str(company_id))
                logger.info(f"Przetwarzanie spółki {ticker} (company_id: {company_id})...")

                try:
                    market_raw = load_market_data(company_id)
                    market_data = _prep_market(market_raw)
                except Exception as e:
                    logger.error(f"Błąd podczas pobierania danych rynkowych dla {ticker}: {e}")
                    continue

                if market_data.empty:
                    logger.info(f"Brak danych rynkowych dla {ticker}, pomijanie...")
                    continue

                fundamentals_raw = load_fundamentals(company_id)
                fundamentals_data = _prep_fundamentals(fundamentals_raw)

                # Build aligned, as-of-merged frame
                final_df = _merge_asof_on_market_dates(market_data, fundamentals_data, macro_data)
                if final_df.empty:
                    logger.info(f"Brak danych do zapisania po scaleniu dla {ticker}")
                    continue

                # Keep only columns that exist in FeaturesFinalPrepared
                # (Adjust this list to match your ORM model exactly)
                allowed_cols = {
                    "company_id", "date",
                    "close",
                    "interest_income","fee_income","banking_result","gross_profit","net_profit",
                    "amortization","assets","equity","shares_thousands","bvps","eps",
                    "gdp","cpi","unemployment_rate","interest_rate","exchange_rate_eur","exchange_rate_usd",
                }
                cols = [c for c in final_df.columns if c in allowed_cols]
                final_df = final_df[cols]

                # Upsert in chunks
                records = [ _nan_to_none_record(r) for r in final_df.to_dict(orient="records") ]
                total = len(records)
                if total == 0:
                    logger.info(f"Brak rekordów do zapisu dla {ticker}")
                    continue

                for i in range(0, total, self.batch_size):
                    chunk = records[i:i+self.batch_size]
                    stmt = insert(FeaturesFinalPrepared).values(chunk)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["company_id", "date"],
                        set_={k: stmt.excluded[k] for k in chunk[0].keys() if k not in ("company_id", "date")}
                    )
                    self.db.execute(stmt)
                self.db.commit()
                logger.info(f"Zapisano/uzupełniono {total} rekordów dla {ticker}")

        except Exception as e:
            logger.error(f"Błąd w pipeline: {e}")
            self.db.rollback()
        finally:
            self.db.close()


if __name__ == "__main__":
    FundamentalsBranchPipeline().run()
