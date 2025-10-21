from fastapi import APIRouter
from app.services.fetchers.fundamentals_fetcher_backup import update_fundamentals_all_companies

router = APIRouter()

@router.post("/fundamentals/update")
def update_fundamentals():
    update_fundamentals_all_companies()
    return {"status": "Dane fundamentals zostały zaktualizowane."}
