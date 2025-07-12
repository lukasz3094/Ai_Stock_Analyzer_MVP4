from app.db.models import Company, NewsArticles, NewsContextLink
from app.core.config import SessionLocal
from app.services.getters.companies_getter import get_all_company_aliases
from app.services.getters.sectors_getter import get_all_context_tags
from sqlalchemy.orm import Session
from multiprocessing import Pool, cpu_count, current_process
from tqdm import tqdm
import spacy
from math import floor

BATCH_SIZE = 500
nlp = None

def init_worker():
    global nlp
    nlp = spacy.load("pl_core_news_lg")

def classify_news_article(news_dict, company_aliases, context_tags):
    global nlp
    text = f"{news_dict['headline']} {news_dict['content']}".lower()
    doc = nlp(text)

    company_matches = []
    for alias in company_aliases:
        if alias['alias'].lower() in text:
            company_matches.append(alias['company_id'])

    tag_scores = {
        tag['id']: doc.similarity(nlp(tag['name'].lower()))
        for tag in context_tags
    }

    best_context_tag_id = max(tag_scores, key=tag_scores.get)
    best_score = tag_scores[best_context_tag_id]

    return {
        "news_article_id": news_dict['id'],
        "company_ids": list(set(company_matches)),
        "context_tag_id": best_context_tag_id,
        "confidence_score": best_score
    }

def load_lookup_data():
    db = SessionLocal()
    try:
        aliases = get_all_company_aliases()
        tags = get_all_context_tags()
        alias_list = [{"company_id": a.company_id, "alias": a.alias} for a in aliases]
        tag_list = [{"id": t.id, "name": t.name} for t in tags]
        return alias_list, tag_list
    finally:
        db.close()

def classify_batch(args):
    news_batch, company_aliases, context_tags = args
    return [
        classify_news_article(news, company_aliases, context_tags)
        for news in news_batch
    ]

def process_unclassified_news():
    all_news = []
    with SessionLocal() as db:
        news_records = db.query(NewsArticles).filter(
            ~NewsArticles.id.in_(db.query(NewsContextLink.news_article_id))
        ).all()
        all_news = [{"id": n.id, "headline": n.headline, "content": n.content} for n in news_records]

    company_aliases, context_tags = load_lookup_data()
    task_args = [
        (all_news[i:i + BATCH_SIZE], company_aliases, context_tags)
        for i in range(0, len(all_news), BATCH_SIZE)
    ]

    with Pool(floor(cpu_count() / 2), initializer=init_worker) as pool:
        for batch_results in tqdm(pool.imap_unordered(classify_batch, task_args), total=len(task_args), desc="Classifying"):
            with SessionLocal() as db:
                for result in batch_results:
                    for company_id in result["company_ids"]:
                        sector_id = db.query(Company).filter(Company.id == company_id).first().sector_id

                        link = NewsContextLink(
                            news_article_id=result["news_article_id"],
                            company_id=company_id,
                            sector_id=sector_id,
                            context_tag_id=result["context_tag_id"],
                            confidence_score=result["confidence_score"]
                        )
                        db.add(link)
                db.commit()
