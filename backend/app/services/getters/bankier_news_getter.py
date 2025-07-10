from app.core import logger
from app.core.config import SessionLocal
from app.db.models import NewsArticles

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