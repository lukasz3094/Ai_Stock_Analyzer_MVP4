# ---------------------- for development purposes only ----------------------

# import requests
# from bs4 import BeautifulSoup
# from datetime import datetime
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm
# from app.core.logger import logger
# from app.db.models import NewsArticles
# from app.core.config import SessionLocal

# class BankierNewsFetcher:
#     BASE_URL = "https://www.bankier.pl"

#     def fetch_article(self, url: str):
#         try:
#             resp = requests.get(url, timeout=10)
#             soup = BeautifulSoup(resp.text, "html.parser")

#             date_tag = soup.find("meta", {"property": "article:published_time"})
#             if not date_tag:
#                 return None
#             date = datetime.fromisoformat(date_tag["content"])

#             title_tag = soup.find("meta", {"property": "og:title"})
#             title = title_tag["content"] if title_tag else ""

#             section = soup.find("section", class_="o-article-content")
#             if not section:
#                 return None
            
#             paragraphs = section.find_all("p")

#             content = ""

#             if paragraphs:
#                 content = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
#             else:
#                 content = section.get_text(strip=True)

#             return NewsArticles(
#                 date=date,
#                 headline=title,
#                 content=content,
#                 url=url,
#                 source="Bankier.pl"
#             )
#         except Exception as e:
#             logger.error(f"[ARTICLE ERROR] {url} -> {e}")
#             return None

#     def fetch_page(self, page: int, from_date: datetime):
#         url = f"{self.BASE_URL}/gielda/wiadomosci/{page}"
#         logger.info(f"[PAGE {page}] Fetching page")

#         try:
#             response = requests.get(url, timeout=10)
#             soup = BeautifulSoup(response.text, "html.parser")
#         except Exception as e:
#             logger.error(f"[PAGE {page}] Error: {e}")
#             return []

#         articles = soup.select("ul.m-listing-article-list li a.m-listing-article-list__anchor")
#         if not articles:
#             return []

#         db = SessionLocal()
#         saved_articles = []

#         for a in articles:
#             href = a.get("href")
#             if not href:
#                 continue

#             article_url = href if href.startswith("http") else f"{self.BASE_URL}{href}"
#             article = self.fetch_article(article_url)

#             if not article:
#                 continue

#             if article.date < from_date:
#                 db.close()
#                 return "STOP"

#             existing = db.query(NewsArticles).filter_by(url=article.url).first()

#             if existing:
#                 if existing.content not in [None, ""]:
#                     continue

#                 existing.content = article.content
#                 existing.headline = article.headline or existing.headline
#                 existing.date = article.date or existing.date
#                 try:
#                     db.commit()
#                     logger.info(f"[PAGE {page}] Updated empty content: {article.url}")
#                 except Exception as e:
#                     db.rollback()
#                     logger.error(f"[PAGE {page}] Update error: {e}")
#             else:
#                 try:
#                     db.add(article)
#                     db.commit()
#                     saved_articles.append(article.url)
#                     logger.info(f"[PAGE {page}] Saved: {article.url}")
#                 except Exception as e:
#                     db.rollback()
#                     logger.error(f"[PAGE {page}] Save error: {e}")

#             time.sleep(0.2)

#         db.close()
#         return saved_articles

#     def fetch_and_update_parallel(self, from_date: datetime, max_pages: int = 12000, max_workers: int = 8):
#         logger.info(f"Starting parallel fetch with {max_workers} workers")
#         stop_flag = False
#         with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=max_pages, unit="page") as pbar:
#             futures = {
#                 executor.submit(self.fetch_page, page, from_date): page
#                 for page in range(1, max_pages + 1)
#             }

#             for future in as_completed(futures):
#                 page = futures[future]
#                 try:
#                     result = future.result()
#                     if result == "STOP":
#                         logger.info(f"[PAGE {page}] Stop condition met. Ending early.")
#                         break
#                 except Exception as e:
#                     logger.error(f"[PAGE {page}] Unhandled error: {e}")
#                 pbar.update(1)

# if __name__ == "__main__":
#     from_date = datetime(2002, 1, 1)
#     fetcher = BankierNewsFetcher()
#     fetcher.fetch_and_update_parallel(from_date, max_pages=12000, max_workers=32)

# ---------------------- Correct one ----------------------
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from app.core.logger import logger
from app.db.models import NewsArticles
from app.core.config import SessionLocal

class BankierNewsFetcher:
    BASE_URL = "https://www.bankier.pl"

    def fetch_article(self, url: str):
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")

            date_tag = soup.find("meta", {"property": "article:published_time"})
            if not date_tag:
                return None
            date = datetime.fromisoformat(date_tag["content"])

            title_tag = soup.find("meta", {"property": "og:title"})
            title = title_tag["content"] if title_tag else ""

            section = soup.find("section", class_="o-article-content")
            if not section:
                return None
            
            paragraphs = section.find_all("p")
            content = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)) if paragraphs else section.get_text(strip=True)

            return NewsArticles(
                date=date,
                headline=title,
                content=content,
                url=url,
                source="Bankier.pl"
            )
        except Exception as e:
            logger.error(f"[ARTICLE ERROR] {url} -> {e}")
            return None

    def fetch_page(self, page: int, from_date: datetime):
        url = f"{self.BASE_URL}/gielda/wiadomosci/{page}"
        logger.info(f"[PAGE {page}] Fetching page")

        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception as e:
            logger.error(f"[PAGE {page}] Error: {e}")
            return []

        articles = soup.select("ul.m-listing-article-list li a.m-listing-article-list__anchor")
        if not articles:
            return []

        article_urls = []
        for a in articles:
            href = a.get("href")
            if not href:
                continue
            article_url = href if href.startswith("http") else f"{self.BASE_URL}{href}"
            article_urls.append(article_url)

        # Optional: Use threading to fetch articles in parallel
        fetched_articles = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self.fetch_article, url): url for url in article_urls}
            for future in as_completed(futures):
                article = future.result()
                if article:
                    fetched_articles.append(article)

        db = SessionLocal()
        for article in sorted(fetched_articles, key=lambda a: a.date, reverse=True):
            if article.date < from_date:
                db.close()
                return "STOP"

            existing = db.query(NewsArticles).filter_by(url=article.url).first()
            if existing:
                if existing.content not in [None, ""]:
                    continue
                existing.content = article.content
                existing.headline = article.headline or existing.headline
                existing.date = article.date or existing.date
                try:
                    db.commit()
                    logger.info(f"[PAGE {page}] Updated empty content: {article.url}")
                except Exception as e:
                    db.rollback()
                    logger.error(f"[PAGE {page}] Update error: {e}")
            else:
                try:
                    db.add(article)
                    db.commit()
                    logger.info(f"[PAGE {page}] Saved: {article.url}")
                except Exception as e:
                    db.rollback()
                    logger.error(f"[PAGE {page}] Save error: {e}")
            time.sleep(0.2)
        db.close()
        return [a.url for a in fetched_articles]

    def fetch_and_update(self, from_date: datetime, max_pages: int = 100):
        logger.info("Starting sequential fetch")
        for page in tqdm(range(1, max_pages + 1), unit="page"):
            result = self.fetch_page(page, from_date)
            if result == "STOP":
                logger.info(f"[PAGE {page}] Stop condition met. Ending early.")
                break

if __name__ == "__main__":
    from_date = datetime.now() - timedelta(hours=1)  # fallback if DB is empty

    db = SessionLocal()
    latest_article = db.query(NewsArticles).order_by(NewsArticles.date.desc()).first()
    if latest_article:
        from_date = latest_article.date
        logger.info(f"Using last article date from DB: {from_date}")
    else:
        logger.warning("No articles in DB. Using fallback from_date.")
    db.close()

    fetcher = BankierNewsFetcher()
    fetcher.fetch_and_update(from_date=from_date, max_pages=100)
