from app.db.models import Sector, ContextTag
from app.core import logger
from app.core.config import SessionLocal


def get_all_sectors():
    db = SessionLocal()
    try:
        sectors = db.query(Sector).all()
        return sectors
    except Exception as e:
        logger.error(f"Error fetching sectors: {e}")
        return []
    finally:
        db.close()

def get_sector_by_id(sector_id):
    db = SessionLocal()
    try:
        sector = db.query(Sector).filter(Sector.id == sector_id).first()
        return sector
    except Exception as e:
        logger.error(f"Error fetching sector with ID {sector_id}: {e}")
        return None
    finally:
        db.close()

def get_all_context_tags():
    db = SessionLocal()
    try:
        tags = db.query(ContextTag).all()
        return tags
    except Exception as e:
        logger.error(f"Error fetching context tags: {e}")
        return []
    finally:
        db.close()

def get_context_tag_by_id(tag_id):
    db = SessionLocal()
    try:
        tag = db.query(ContextTag).filter(ContextTag.id == tag_id).first()
        return tag
    except Exception as e:
        logger.error(f"Error fetching context tag with ID {tag_id}: {e}")
        return None
    finally:
        db.close()
