from app.branches.fundamentals_branch.extractor import (
    load_all_companies,
    load_fundamentals
)
from app.branches.fundamentals_branch.transformer import transform_fundamentals
from app.db.models import FundamentalsFeaturesPrepared
from app.core.config import SessionLocal
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
from app.core.logger import logger

class FundamentalsBranchPipeline:
    def __init__(self):
        self.db = SessionLocal()

    def run(self):
        companies = load_all_companies()

        for _, row in companies.iterrows():
            company_id = row["id"]
            ticker = row["ticker"]
            logger.info(f"Przetwarzanie spółki {ticker}...")

            raw = load_fundamentals(company_id)
            transformed = transform_fundamentals(raw)

            if transformed.empty:
                logger.info(f"Brak danych do zapisania dla {ticker}")
                continue

            for _, row in transformed.iterrows():
                stmt = insert(FundamentalsFeaturesPrepared).values(**row.to_dict())
                stmt = stmt.on_conflict_do_update(
                    index_elements=["company_id", "date"],
                    set_={col: stmt.excluded[col] for col in row.to_dict() if col not in ["company_id", "date"]}
                )
                self.db.execute(stmt)

            self.db.commit()
            logger.info(f"Zapisano {len(transformed)} rekordów dla {ticker}")

        self.db.close()

if __name__ == "__main__":
    pipeline = FundamentalsBranchPipeline()
    pipeline.run()
    logger.info("Pipeline Fundamentals Branch zakończony pomyślnie.")
