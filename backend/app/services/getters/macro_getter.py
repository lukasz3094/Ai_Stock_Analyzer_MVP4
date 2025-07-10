from app.db.models import MacroData
from app.core import logger
from app.core.config import SessionLocal


def get_all_macros():
    db = SessionLocal()
    
    try:
        macros = db.query(MacroData).all()
        return macros
    except Exception as e:
        logger.error(f"Error fetching macros: {e}")
        return []
    finally:
        db.close()

def get_macro_by_date(date):
    db = SessionLocal()
    
    try:
        macro = db.query(MacroData).filter(MacroData.date == date).all()
        return macro
    except Exception as e:
        logger.error(f"Error fetching macro data for date {date}: {e}")
        return []
    finally:
        db.close()

def get_macros_from_date(from_date):
    db = SessionLocal()
    
    try:
        macros = db.query(MacroData).filter(MacroData.date >= from_date).all()
        return macros
    except Exception as e:
        logger.error(f"Error fetching macros from date {from_date}: {e}")
        return []
    finally:
        db.close()