from backend.app.core.config import SessionLocal
from backend.app.db.models import SelectedFeatures


def get_selected_features(sector_id: int) -> list[str]:
    db = SessionLocal()
    latest = (
        db.query(SelectedFeatures)
        .filter_by(sector_id=sector_id, model_type="boruta")
        .order_by(SelectedFeatures.run_date.desc())
        .first()
    )
    db.close()
    return latest.selected_features if latest else []
