from fastapi import APIRouter
from app.services.fetchers.ohlcv_fetcher import update_ohlcv_all_companies

router = APIRouter()

@router.post("/ohlcv/update")
def update_ohlcv():
    update_ohlcv_all_companies()
    return {"status": "Dane OHLCV zostały zaktualizowane."}
