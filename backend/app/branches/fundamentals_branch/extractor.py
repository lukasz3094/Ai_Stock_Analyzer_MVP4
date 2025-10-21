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

def load_fundamentals(company_id: int) -> pd.DataFrame:
    with get_db() as db:
        rows = db.query(Fundamentals).filter(
            Fundamentals.company_id == company_id
        ).order_by(Fundamentals.date).all()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame([{
            "company_id": r.company_id,
            "date": r.date,
            "interest_income": r.interest_income,
            "fee_income": r.fee_income,
            "banking_result": r.banking_result,
            "gross_profit": r.gross_profit,
            "net_profit": r.net_profit,
            "amortization": r.amortization,
            "assets": r.assets,
            "equity": r.equity,
            "shares_thousands": r.shares_thousands,
            "bvps": r.bvps,
            "eps": r.eps
        } for r in rows])

def load_all_companies() -> pd.DataFrame:
    with get_db() as db:
        companies = db.query(Company.id, Company.ticker).all()
        return pd.DataFrame(companies, columns=["id", "ticker"])
