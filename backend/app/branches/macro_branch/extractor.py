from app.core.config import SessionLocal
from app.db.models import MacroData
import pandas as pd

def load_macro_data() -> pd.DataFrame:
    db = SessionLocal()
    try:
        records = db.query(MacroData).order_by(MacroData.date).all()
        return pd.DataFrame([{
            "date": r.date,
            "gdp": r.gdp,
            "cpi": r.cpi,
            "unemployment_rate": r.unemployment_rate,
            "interest_rate": r.interest_rate,
            "exchange_rate_eur": r.exchange_rate_eur,
            "exchange_rate_usd": r.exchange_rate_usd,
        } for r in records])
    finally:
        db.close()
