from multiprocessing import get_context
from tqdm import tqdm
from app.db.models import Company, NewsArticles, NewsFeaturesPrepared
from app.core.config import SessionLocal
from app.core.logger import logger
from app.branches.news_branch.classifier import classify_batch_in_process, init_worker
import gc


def serialize_article(article):
    return {
        "id": article.id,
        "headline": article.headline,
        "content": article.content,
        "date": article.date.isoformat() if article.date else None,
        "url": article.url,
        "source": article.source,
    }

class NewsBranchPipeline:
    def __init__(self):
        self.db = SessionLocal()
        self.pool = get_context("spawn").Pool(processes=6, initializer=init_worker)

    def load_raw_data(self):
        chunk_size = 20000
        subquery = self.db.query(NewsFeaturesPrepared.news_article_id).distinct()
        total = self.db.query(NewsArticles).filter(~NewsArticles.id.in_(subquery)).count()

        self.raw_news = []
        for offset in tqdm(range(0, total, chunk_size), desc="Loading news"):
            chunk = self.db.query(NewsArticles).filter(
                ~NewsArticles.id.in_(subquery)
            ).offset(offset).limit(chunk_size).all()
            self.raw_news.extend([serialize_article(a) for a in chunk])

    def transform(self, news_articles):
        batch_size = 50

        for i in tqdm(range(0, len(news_articles), batch_size), desc="Łączny postęp przetwarzania"):
            batch = news_articles[i:i + batch_size]
            result = self.pool.map(classify_batch_in_process, [batch])[0]
            self.save_classified(result)
            gc.collect()

    def save_classified(self, classified_batch):
        chunk_size = 5000
        to_insert = []

        existing_pairs = {
            (row.news_article_id, row.company_id)
            for row in self.db.query(NewsFeaturesPrepared.news_article_id, NewsFeaturesPrepared.company_id)
        }

        company_sector_map = {
            c.id: c.sector_id
            for c in self.db.query(Company.id, Company.sector_id)
        }

        for result in classified_batch:
            for company_id in result["company_ids"]:
                key = (result["news_article_id"], company_id)
                if key in existing_pairs:
                    continue

                sector_id = company_sector_map.get(company_id)
                to_insert.append(NewsFeaturesPrepared(
                    news_article_id=result["news_article_id"],
                    company_id=company_id,
                    sector_id=sector_id,
                    context_tag_id=result["context_tag_id"],
                    confidence_score=result["confidence_score"],
                    sentiment_score=result["sentiment_score"],
                    sentiment_label=result["sentiment_label"]
                ))
                existing_pairs.add(key)

        for i in range(0, len(to_insert), chunk_size):
            self.db.bulk_save_objects(to_insert[i:i + chunk_size])
            self.db.commit()

    def run(self):
        try:
            self.load_raw_data()
            self.transform(self.raw_news)
            print("Pipeline zakończony sukcesem.")
        except Exception as e:
            print(f"Błąd w pipeline: {e}")
        finally:
            self.db.close()


if __name__ == "__main__":
    pipeline = NewsBranchPipeline()
    pipeline.run()
    print("Pipeline news_branch zakończony.")
