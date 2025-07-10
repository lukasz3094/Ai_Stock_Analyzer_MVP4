from app.db.models import MarketFeatures
from app.core import logger
from app.core.config import SessionLocal

def get_all_features():
    db = SessionLocal()
    
    try:
        features = db.query(MarketFeatures).all()
        return features
    except Exception as e:
        logger.error(f"Error fetching features: {e}")
        return []
    finally:
        db.close()

def get_features_by_company(company_id):
    db = SessionLocal()
    
    try:
        features = db.query(MarketFeatures).filter(MarketFeatures.company_id == company_id).all()
        return features
    except Exception as e:
        logger.error(f"Error fetching features for company {company_id}: {e}")
        return []
    finally:
        db.close()

def get_features_by_date(date):
    db = SessionLocal()
    
    try:
        features = db.query(MarketFeatures).filter(MarketFeatures.date == date).all()
        return features
    except Exception as e:
        logger.error(f"Error fetching features for date {date}: {e}")
        return []
    finally:
        db.close()

def get_features_by_company_and_date(company_id, date):
    db = SessionLocal()
    
    try:
        features = db.query(MarketFeatures).filter(
            MarketFeatures.company_id == company_id,
            MarketFeatures.date == date
        ).all()
        return features
    except Exception as e:
        logger.error(f"Error fetching features for company {company_id} on date {date}: {e}")
        return []
    finally:
        db.close()

def get_features_from_date(from_date):
    db = SessionLocal()
    
    try:
        features = db.query(MarketFeatures).filter(MarketFeatures.date >= from_date).all()
        return features
    except Exception as e:
        logger.error(f"Error fetching features from date {from_date}: {e}")
        return []
    finally:
        db.close()

def get_features_by_company_from_date(company_id, from_date):
    db = SessionLocal()
    
    try:
        features = db.query(MarketFeatures).filter(
            MarketFeatures.company_id == company_id,
            MarketFeatures.date >= from_date
        ).all()
        return features
    except Exception as e:
        logger.error(f"Error fetching features for company {company_id} from date {from_date}: {e}")
        return []
    finally:
        db.close()