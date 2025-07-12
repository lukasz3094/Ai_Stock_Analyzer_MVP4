from fastapi import APIRouter
from app.services.getters.bankier_news_getter import (get_all_bankier_news, get_bankier_news_by_date, 
                                                      get_bankier_news_from_date, get_all_classified_news,
                                                      get_classified_news_by_company, get_classified_news_by_context_tag,
                                                      get_classified_news_by_sector, get_all_verified_classified_news,
                                                      get_verified_classified_news_by_company, get_verified_classified_news_by_sector,
                                                      get_verified_classified_news_by_context_tag)

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

@router.get("/classified/news_articles/all")
def all_classified_news_articles():
    classified_news = get_all_classified_news()
    return {"classified_news_articles": classified_news}

@router.get("/classified/news_articles/company/{company_id}")
def classified_news_by_company(company_id: int):
    classified_news = get_classified_news_by_company(company_id)
    return {"classified_news_articles": classified_news}

@router.get("/classified/news_articles/context_tag/{context_tag_id}")
def classified_news_by_context_tag(context_tag_id: int):
    classified_news = get_classified_news_by_context_tag(context_tag_id)
    return {"classified_news_articles": classified_news}

@router.get("/classified/news_articles/sector/{sector_id}")
def classified_news_by_sector(sector_id: int):
    classified_news = get_classified_news_by_sector(sector_id)
    return {"classified_news_articles": classified_news}

@router.get("/verified/classified/news_articles/all")
def all_verified_classified_news_articles():
    verified_classified_news = get_all_verified_classified_news()
    return {"verified_classified_news_articles": verified_classified_news}

@router.get("/verified/classified/news_articles/company/{company_id}")
def verified_classified_news_by_company(company_id: int):
    verified_classified_news = get_verified_classified_news_by_company(company_id)
    return {"verified_classified_news_articles": verified_classified_news}

@router.get("/verified/classified/news_articles/context_tag/{context_tag_id}")
def verified_classified_news_by_context_tag(context_tag_id: int):
    verified_classified_news = get_verified_classified_news_by_context_tag(context_tag_id)
    return {"verified_classified_news_articles": verified_classified_news}

@router.get("/verified/classified/news_articles/sector/{sector_id}")
def verified_classified_news_by_sector(sector_id: int):
    verified_classified_news = get_verified_classified_news_by_sector(sector_id)
    return {"verified_classified_news_articles": verified_classified_news}
