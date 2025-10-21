from app.db.models import FeaturesFinalPrepared
from app.core import logger
from app.core.config import SessionLocal

def get_features_for_company(company_id: int) -> list[FeaturesFinalPrepared]:
    db = SessionLocal()
    
    try:
        features = db.query(FeaturesFinalPrepared).filter(
            FeaturesFinalPrepared.company_id == company_id
        ).all()

        return features
    except Exception as e:
        logger.error(f"Error fetching features for company {company_id}: {e}")
        return []
    finally:
        db.close()
