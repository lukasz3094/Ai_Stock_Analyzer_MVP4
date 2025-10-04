from app.core.logger import logger
from app.db.models import GpwSessions
from app.core.config import SessionLocal


def get_all_gpwsessions():
    db = SessionLocal()
    
    try:
        gpw_sessions = db.query(GpwSessions).all()
        return gpw_sessions
    except Exception as e:
        logger.error(f"Error fetching GPW sessions: {e}")
        return []
    finally:
        db.close()

def get_gpwsessions_by_range(start_date, end_date):
    db = SessionLocal()
    
    try:
        gpw_sessions = db.query(GpwSessions).filter(GpwSessions.trade_date.between(start_date, end_date)).all()
        return gpw_sessions
    except Exception as e:
        logger.error(f"Error fetching GPW sessions for date range {start_date} - {end_date}: {e}")
        return []
    finally:
        db.close()

def get_gpwsessions_by_start_date(date):
    db = SessionLocal()

    try:
        gpw_sessions = db.query(GpwSessions).filter(GpwSessions.trade_date >= date).all()
        return gpw_sessions
    except Exception as e:
        logger.error(f"Error fetching GPW sessions for start date {date}: {e}")
        return []
    finally:
        db.close()

def get_gpwsessions_by_end_date(date):
    db = SessionLocal()

    try:
        gpw_sessions = db.query(GpwSessions).filter(GpwSessions.trade_date <= date).all()
        return gpw_sessions
    except Exception as e:
        logger.error(f"Error fetching GPW sessions for end date {date}: {e}")
        return []
    finally:
        db.close()