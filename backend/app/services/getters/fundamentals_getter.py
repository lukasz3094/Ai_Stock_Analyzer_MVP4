from app.core import logger
from app.db.models import Fundamentals
from app.core.config import SessionLocal


def get_all_fundamentals():
    db = SessionLocal()
    
    try:
        fundamentals = db.query(Fundamentals).all()
        return fundamentals
    except Exception as e:
        logger.error(f"Error fetching fundamentals: {e}")
        return []
    finally:
        db.close()

def get_fundamentals_by_company(company_id):
    db = SessionLocal()
    
    try:
        fundamentals = db.query(Fundamentals).filter(Fundamentals.company_id == company_id).all()
        return fundamentals
    except Exception as e:
        logger.error(f"Error fetching fundamentals for company {company_id}: {e}")
        return []
    finally:
        db.close()

def get_fundamentals_by_date(date):
    db = SessionLocal()
    
    try:
        fundamentals = db.query(Fundamentals).filter(Fundamentals.date == date).all()
        return fundamentals
    except Exception as e:
        logger.error(f"Error fetching fundamentals for date {date}: {e}")
        return []
    finally:
        db.close()

def get_fundamentals_by_company_and_date(company_id, date):
    db = SessionLocal()
    
    try:
        fundamentals = db.query(Fundamentals).filter(
            Fundamentals.company_id == company_id,
            Fundamentals.date == date
        ).all()
        return fundamentals
    except Exception as e:
        logger.error(f"Error fetching fundamentals for company {company_id} on date {date}: {e}")
        return []
    finally:
        db.close()

def get_fundamentals_from_date(from_date):
    db = SessionLocal()
    
    try:
        fundamentals = db.query(Fundamentals).filter(Fundamentals.date >= from_date).all()
        return fundamentals
    except Exception as e:
        logger.error(f"Error fetching fundamentals from date {from_date}: {e}")
        return []
    finally:
        db.close()

def get_fundamentals_by_company_from_date(company_id, start_date):
    db = SessionLocal()
    try:
        fundamentals = db.query(Fundamentals).filter(
            Fundamentals.company_id == company_id,
            Fundamentals.date >= start_date
        ).all()
        return fundamentals
    except Exception as e:
        logger.error(f"Error fetching fundamentals for company {company_id} from {start_date}: {e}")
        return []
    finally:
        db.close()