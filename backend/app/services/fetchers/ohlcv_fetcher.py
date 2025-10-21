import pandas as pd
import requests
import io
from datetime import timedelta
from sqlalchemy.exc import SQLAlchemyError
from app.core.config import SessionLocal
from app.db.models import Company, MarketData
from app.core.logger import logger
from app.services.getters.companies_getter import get_companies_by_group
import yfinance as yf


def fetch_yahoo_ohlcv(ticker_yahoo: str) -> pd.DataFrame:
    ticker_yahoo = ticker_yahoo + ".WA"
    df = yf.download(ticker_yahoo, period="max", interval="1d", auto_adjust=True)

    if df.empty:
        logger.error(f"[{ticker_yahoo}]: brak danych OHLCV z Yahoo Finance.")
        return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df['Date'] = df['Date'].astype(str)

    clean_data = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    clean_data = clean_data.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    return clean_data

def update_ohlcv_all_companies():
    db = SessionLocal()
    companies = db.query(Company).all()

    logger.info(f"Rozpoczynam aktualizację danych OHLCV dla {len(companies)} spółek.")

    i = 1
    number_of_companies = len(companies)

    for company in companies:
        logger.info(f"Przetwarzanie spółki {i} z {number_of_companies}: {company.ticker}")

        i += 1

        try:
            df = fetch_yahoo_ohlcv(company.ticker)

            if df is None or df.empty:
                logger.error(f"[{company.ticker}]: brak danych OHLCV do aktualizacji.")
                continue

            # najnowsza data w bazie
            last_record = (
                db.query(MarketData)
                .filter_by(company_id=company.id)
                .order_by(MarketData.date.desc())
                .first()
            )

            if last_record:
                df = df[pd.to_datetime(df["date"]).dt.date > last_record.date]

            if df.empty:
                logger.info(f"[{company.ticker}]: brak nowych danych do aktualizacji.")
                continue

            start_date = pd.to_datetime(df["date"].min())
            logger.info(f"[{company.ticker}]: aktualizacja danych od {start_date.date()}.")

            for _, row in df.iterrows():
                entry = MarketData(
                    company_id=company.id,
                    date=row["date"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=int(row["volume"]) if not pd.isna(row["volume"]) else 0,
                )
                db.merge(entry)

            db.commit()
            logger.info(f"[{company.ticker}]: dane OHLCV zaktualizowane.")

        except (requests.RequestException, SQLAlchemyError) as e:
            db.rollback()
            logger.error(f"[{company.ticker}]: błąd podczas aktualizacji danych OHLCV: {e}")

    db.close()

def update_ohlcv_selected_companies(company_ids):
    db = SessionLocal()
    companies = db.query(Company).filter(Company.id.in_(company_ids)).all()

    logger.info(f"Rozpoczynam aktualizację danych OHLCV dla {len(companies)} spółek.")

    i = 1
    number_of_companies = len(companies)

    for company in companies:
        logger.info(f"Przetwarzanie spółki {i} z {number_of_companies}: {company.ticker}")

        i += 1

        try:
            df = fetch_yahoo_ohlcv(company.ticker)

            if df is None or df.empty:
                logger.error(f"[{company.ticker}]: brak danych OHLCV do aktualizacji.")
                continue

            # najnowsza data w bazie
            last_record = (
                db.query(MarketData)
                .filter_by(company_id=company.id)
                .order_by(MarketData.date.desc())
                .first()
            )

            if last_record:
                df = df[pd.to_datetime(df["date"]).dt.date > last_record.date]

            if df.empty:
                logger.info(f"[{company.ticker}]: brak nowych danych do aktualizacji.")
                continue

            start_date = pd.to_datetime(df["date"].min())
            logger.info(f"[{company.ticker}]: aktualizacja danych od {start_date.date()}.")

            for _, row in df.iterrows():
                entry = MarketData(
                    company_id=company.id,
                    date=row["date"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=int(row["volume"]) if not pd.isna(row["volume"]) else 0,
                )
                db.merge(entry)

            db.commit()
            logger.info(f"[{company.ticker}]: dane OHLCV zaktualizowane.")

        except (requests.RequestException, SQLAlchemyError) as e:
            db.rollback()
            logger.error(f"[{company.ticker}]: błąd podczas aktualizacji danych OHLCV: {e}")

    db.close()

if __name__ == "__main__":
    companies = get_companies_by_group("wig-banki")
    company_ids = [company.id for company in companies]
    update_ohlcv_selected_companies(company_ids)
