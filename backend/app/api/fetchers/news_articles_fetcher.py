from datetime import datetime
from fastapi import APIRouter
from app.services.fetchers.bankier_news_fetcher import BankierNewsFetcher
router = APIRouter()

@router.post("/news-articles/update/bankier")
def update_bankier_news():
    from_date = datetime(2002, 1, 1)
    fetcher = BankierNewsFetcher()
    fetcher.fetch_and_update(from_date)
    return {"status": "Dane artykułów zostały zaktualizowane."}