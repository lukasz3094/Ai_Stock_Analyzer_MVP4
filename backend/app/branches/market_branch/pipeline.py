from app.branches.market_branch.extractor import (
    load_all_companies,
    load_market_data,
    load_existing_feature_keys
)
from app.branches.market_branch.transformer import transform_market_data
from app.db.models import MarketFeaturesPrepared
from app.core.config import SessionLocal
from sqlalchemy.dialects.postgresql import insert
from app.core.logger import logger


class MarketBranchPipeline:
    def __init__(self):
        self.db = SessionLocal()
        self.existing_keys = load_existing_feature_keys()

    def run(self):
        companies = load_all_companies()

        for _, row in companies.iterrows():
            company_id = row["id"]
            ticker = row["ticker"]
            logger.info(f"Przetwarzanie danych rynkowych dla {ticker}...")

            raw = load_market_data(company_id)
            transformed = transform_market_data(raw, self.existing_keys)

            if transformed.empty:
                logger.info(f"Brak nowych danych rynkowych dla {ticker}")
                continue

            for _, row in transformed.iterrows():
                stmt = insert(MarketFeaturesPrepared).values(**row.to_dict())
                stmt = stmt.on_conflict_do_nothing(index_elements=["company_id", "date"])
                self.db.execute(stmt)

            self.db.commit()
            logger.info(f"Zapisano {len(transformed)} rekordów rynkowych dla {ticker}")

        self.db.close()


if __name__ == "__main__":
    MarketBranchPipeline().run()
    logger.info("Pipeline market_branch zakończony.")
