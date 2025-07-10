from fastapi import APIRouter
from app.services.fetchers.features_fetcher import update_features_all_companies
router = APIRouter()

@router.post("/features/update")
def update_features():
    update_features_all_companies()
    return {"status": "Dane features zostały zaktualizowane."}
