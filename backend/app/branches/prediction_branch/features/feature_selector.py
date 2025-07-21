from app.db.models import SelectedFeatures
from app.core.config import SessionLocal
from datetime import date

def store_boruta_result(sector_id: int, feature_names: list[str], importances: dict[str, float]):
    db = SessionLocal()
    record = SelectedFeatures(
        sector_id=sector_id,
        run_date=date.today(),
        model_type="boruta",
        selected_features=feature_names,
        total_features=len(feature_names),
        feature_importances=importances
    )
    db.add(record)
    db.commit()
    db.close()
