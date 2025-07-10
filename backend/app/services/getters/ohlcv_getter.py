from app.db.models import MarketData
from app.core import logger
from app.core.config import SessionLocal


def get_all_ohlcv():
    db = SessionLocal()
    
    try:
        ohlcv_data = db.query(MarketData).all()
        return ohlcv_data
    except Exception as e:
        logger.error(f"Error fetching OHLCV data: {e}")
        return []
    finally:
        db.close()


def get_ohlcv_by_date(date):
    db = SessionLocal()
    
    try:
        ohlcv_data = db.query(MarketData).filter(MarketData.date == date).all()
        return ohlcv_data
    except Exception as e:
        logger.error(f"Error fetching OHLCV data for date {date}: {e}")
        return []
    finally:
        db.close()

def get_ohlcv_from_date(from_date):
    db = SessionLocal()
    
    try:
        ohlcv_data = db.query(MarketData).filter(MarketData.date >= from_date).all()
        return ohlcv_data
    except Exception as e:
        logger.error(f"Error fetching OHLCV data from date {from_date}: {e}")
        return []
    finally:
        db.close()

def get_ohlcv_by_company(company_id):
    db = SessionLocal()
    
    try:
        ohlcv_data = db.query(MarketData).filter(MarketData.company_id == company_id).all()
        return ohlcv_data
    except Exception as e:
        logger.error(f"Error fetching OHLCV data for company {company_id}: {e}")
        return []
    finally:
        db.close()

def get_ohlcv_by_company_and_date(company_id, date):
    db = SessionLocal()
    
    try:
        ohlcv_data = db.query(MarketData).filter(
            MarketData.company_id == company_id,
            MarketData.date == date
        ).all()
        return ohlcv_data
    except Exception as e:
        logger.error(f"Error fetching OHLCV data for company {company_id} on date {date}: {e}")
        return []
    finally:
        db.close()

def get_ohlcv_by_company_and_from_date(company_id, from_date):
    db = SessionLocal()
    
    try:
        ohlcv_data = db.query(MarketData).filter(
            MarketData.company_id == company_id,
            MarketData.date >= from_date
        ).all()
        return ohlcv_data
    except Exception as e:
        logger.error(f"Error fetching OHLCV data for company {company_id} from date {from_date}: {e}")
        return []
    finally:
        db.close()

