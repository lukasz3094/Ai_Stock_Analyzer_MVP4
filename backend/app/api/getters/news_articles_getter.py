from fastapi import APIRouter
from app.services.getters.bankier_news_getter import get_all_bankier_news, get_bankier_news_by_date, get_bankier_news_from_date

router = APIRouter()


@router.get("/news_articles/all")
def all_news_articles():
    news_articles = get_all_bankier_news()
    return {"news_articles": news_articles}

@router.get("/news_articles/date/{date}")
def news_articles_by_date(date: str):
    news_articles = get_bankier_news_by_date(date)
    return {"news_articles": news_articles}

@router.get("/news_articles/from_date/{from_date}")
def news_articles_from_date(from_date: str):
    news_articles = get_bankier_news_from_date(from_date)
    return {"news_articles": news_articles}
    