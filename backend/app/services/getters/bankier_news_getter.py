from app.core import logger
from app.core.config import SessionLocal
from app.db.models import NewsArticles, NewsContextLink

def get_all_bankier_news():
    db = SessionLocal()
    try:
        news = db.query(NewsArticles).filter(NewsArticles.source == "Bankier.pl").all()
        return news
    except Exception as e:
        logger.error(f"Error fetching Bankier news: {e}")
        return []
    finally:
        db.close()
    
def get_bankier_news_by_date(date):
    db = SessionLocal()
    try:
        news = db.query(NewsArticles).filter(
            NewsArticles.source == "Bankier.pl",
            NewsArticles.date == date
        ).all()
        return news
    except Exception as e:
        logger.error(f"Error fetching Bankier news for date {date}: {e}")
        return []
    finally:
        db.close()
    
def get_bankier_news_from_date(from_date):
    db = SessionLocal()
    try:
        news = db.query(NewsArticles).filter(
            NewsArticles.source == "Bankier.pl",
            NewsArticles.date >= from_date
        ).all()
        return news
    except Exception as e:
        logger.error(f"Error fetching Bankier news from date {from_date}: {e}")
        return []
    finally:
        db.close()

def get_all_classified_news():
    db = SessionLocal()
    try:
        news = db.query(NewsContextLink).all()
        return news
    except Exception as e:
        logger.error(f"Error fetching classified Bankier news: {e}")
        return []
    finally:
        db.close()

def get_classified_news_by_company(company_id):
    db = SessionLocal()
    try:
        news = db.query(NewsArticles).join(
            NewsContextLink, NewsArticles.id == NewsContextLink.news_article_id
        ).filter(
            NewsContextLink.company_id == company_id
        ).all()
        return news
    except Exception as e:
        logger.error(f"Error fetching news for company {company_id}: {e}")
        return []
    finally:
        db.close()

def get_classified_news_by_context_tag(context_tag_id):
    db = SessionLocal()
    try:
        news = db.query(NewsArticles).join(
            NewsContextLink, NewsArticles.id == NewsContextLink.news_article_id
        ).filter(
            NewsContextLink.context_tag_id == context_tag_id
        ).all()
        return news
    except Exception as e:
        logger.error(f"Error fetching news for context tag {context_tag_id}: {e}")
        return []
    finally:
        db.close()

def get_classified_news_by_sector(sector_id):
    db = SessionLocal()
    try:
        news = db.query(NewsArticles).join(
            NewsContextLink, NewsArticles.id == NewsContextLink.news_article_id
        ).filter(
            NewsContextLink.sector_id == sector_id
        ).all()
        return news
    except Exception as e:
        logger.error(f"Error fetching news for sector {sector_id}: {e}")
        return []
    finally:
        db.close()

def get_all_verified_classified_news():
    db = SessionLocal()
    try:
        news = db.query(NewsContextLink).filter(NewsContextLink.confidence_score >= 0.7).all()
        return news
    except Exception as e:
        logger.error(f"Error fetching classified Bankier news: {e}")
        return []
    finally:
        db.close()

def get_verified_classified_news_by_company(company_id):
    db = SessionLocal()
    try:
        news = db.query(NewsArticles).join(
            NewsContextLink, NewsArticles.id == NewsContextLink.news_article_id,
            NewsContextLink.confidence_score >= 0.7
        ).filter(
            NewsContextLink.company_id == company_id
        ).all()
        return news
    except Exception as e:
        logger.error(f"Error fetching news for company {company_id}: {e}")
        return []
    finally:
        db.close()

def get_verified_classified_news_by_context_tag(context_tag_id):
    db = SessionLocal()
    try:
        news = db.query(NewsArticles).join(
            NewsContextLink, NewsArticles.id == NewsContextLink.news_article_id,
            NewsContextLink.confidence_score >= 0.7
        ).filter(
            NewsContextLink.context_tag_id == context_tag_id
        ).all()
        return news
    except Exception as e:
        logger.error(f"Error fetching news for context tag {context_tag_id}: {e}")
        return []
    finally:
        db.close()

def get_verified_classified_news_by_sector(sector_id):
    db = SessionLocal()
    try:
        news = db.query(NewsArticles).join(
            NewsContextLink, NewsArticles.id == NewsContextLink.news_article_id,
            NewsContextLink.confidence_score >= 0.7
        ).filter(
            NewsContextLink.sector_id == sector_id
        ).all()
        return news
    except Exception as e:
        logger.error(f"Error fetching news for sector {sector_id}: {e}")
        return []
    finally:
        db.close()
