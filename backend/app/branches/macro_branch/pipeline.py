from app.branches.macro_branch.extractor import load_macro_data
from app.branches.macro_branch.transformer import transform_macro_data
from app.db.models import MacroFeaturesPrepared
from app.core.config import SessionLocal
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import select
import pandas as pd
from app.core.logger import logger


class MacroBranchPipeline:
    def __init__(self):
        self.db = SessionLocal()
        self.raw_data = pd.DataFrame()
        self.transformed = pd.DataFrame()
        self.existing_dates = set()

    def load_raw_data(self):
        self.raw_data = load_macro_data()

    def load_existing_dates(self):
        result = self.db.execute(select(MacroFeaturesPrepared.date))
        self.existing_dates = set(r[0] for r in result)

    def transform(self):
        df = transform_macro_data(self.raw_data)
        if df.empty:
            self.transformed = pd.DataFrame()
            return

        self.transformed = df[~df["date"].isin(self.existing_dates)]

    def save(self):
        if self.transformed.empty:
            logger.info("No new macro data to save.")
            return

        for _, row in self.transformed.iterrows():
            stmt = insert(MacroFeaturesPrepared).values(**row.to_dict())
            stmt = stmt.on_conflict_do_nothing(index_elements=["date"])
            self.db.execute(stmt)

        self.db.commit()
        logger.info(f"Saved {len(self.transformed)} new macro data records.")

    def run(self):
        self.load_existing_dates()
        self.load_raw_data()
        self.transform()
        self.save()

if __name__ == "__main__":
    pipeline = MacroBranchPipeline()
    pipeline.run()
    logger.info("Macro Branch Pipeline completed successfully.")
