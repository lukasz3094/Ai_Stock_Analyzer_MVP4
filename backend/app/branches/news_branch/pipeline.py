from app.db.models import Company, NewsArticles, NewsFeaturesPrepared
from app.core.config import SessionLocal
from app.branches.news_branch.classifier import classify_news_articles

class NewsBranchPipeline:
    def __init__(self):
        self.db = SessionLocal()
        self.raw_news = []
        self.classified_news = []

    def load_raw_data(self):
        self.raw_news = self.db.query(NewsArticles).filter(
            ~NewsArticles.id.in_(
                self.db.query(NewsFeaturesPrepared.news_article_id)
            )
        ).all()

    def transform(self):
        self.classified_news = classify_news_articles(self.raw_news)

    def save(self):
        for result in self.classified_news:
            for company_id in result["company_ids"]:
                sector_id = self.db.query(Company).filter(Company.id == company_id).first().sector_id
                self.db.add(NewsFeaturesPrepared(
                    news_article_id=result["news_article_id"],
                    company_id=company_id,
                    sector_id=sector_id,
                    context_tag_id=result["context_tag_id"],
                    confidence_score=result["confidence_score"]
                ))
        self.db.commit()

    def run(self):
        self.load_raw_data()
        self.transform()
        self.save()
