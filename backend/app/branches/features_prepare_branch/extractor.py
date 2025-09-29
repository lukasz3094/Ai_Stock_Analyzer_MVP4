import pandas as pd
from app.core.config import SessionLocal
from app.db.models import Fundamentals, Company
from contextlib import contextmanager

@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# def load_companies_by_id(company_ids: list[int]) -> pd.DataFrame:
#     with get_db() as db:
#         companies = db.query(Company.id, Company.ticker, Company.sector_id).filter(Company.id.in_(company_ids)).all()
#         return pd.DataFrame(companies, columns=["id", "ticker", "sector_id"])
