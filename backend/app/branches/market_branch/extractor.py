import pandas as pd
from sqlalchemy import select
from app.core.config import SessionLocal
from app.db.models import MarketData, Company, MarketFeaturesPrepared
from contextlib import contextmanager
from app.core.logger import logger

@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def load_all_companies() -> pd.DataFrame:
    with get_db() as db:
        companies = db.query(Company.id, Company.ticker).all()
        return pd.DataFrame(companies, columns=["id", "ticker"])

def load_market_data(company_id: int) -> pd.DataFrame:
    try:
        with get_db() as db:
            rows = db.query(MarketData).filter(
                MarketData.company_id == company_id
            ).order_by(MarketData.date).all()

            if not rows:
                logger.info(f"Brak danych rynkowych dla company_id: {company_id}")
                return pd.DataFrame()

            df = pd.DataFrame([{
                "company_id": r.company_id,
                "date": r.date,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume
            } for r in rows])
            
            return df
            
    except Exception as e:
        logger.error(f"Błąd podczas ładowania danych rynkowych dla company_id {company_id}: {e}")
        return pd.DataFrame()

def load_existing_feature_keys() -> set:
    with get_db() as db:
        result = db.execute(select(MarketFeaturesPrepared.company_id, MarketFeaturesPrepared.date))
        return set(result.fetchall())
