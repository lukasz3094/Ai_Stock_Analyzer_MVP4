from fastapi import APIRouter
from app.services.operations.news_classifier import process_unclassified_news

router = APIRouter()


@router.post("/news/unclassified/process")
def process_unclassified_news_route():
    process_unclassified_news()
    return {"message": "Unclassified news articles processed successfully."}
