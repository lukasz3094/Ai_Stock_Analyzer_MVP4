from app.services.getters.companies_getter import get_companies_by_group

# from app.branches.features_prepare_branch.extractor import load_companies_by_id

from app.branches.fundamentals_branch.extractor import load_fundamentals
from app.branches.fundamentals_branch.transformer import transform_fundamentals

from app.branches.macro_branch.extractor import load_macro_data
from app.branches.macro_branch.transformer import transform_macro_data

from app.branches.market_branch.extractor import load_market_data

from app.db.models import FeaturesPreparedWig20
from app.core.config import SessionLocal
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
from app.core.logger import logger

class FundamentalsBranchPipeline:
    def __init__(self):
        self.db = SessionLocal()

    def run(self):
        try:
            logger.info(f"Pobieranie spółek z grupy WIG20...")
            companies = get_companies_by_group("wig20")

            logger.info(f"Pobieranie danych makroekonomicznych...")
            macro_data = load_macro_data()
            logger.info(f"Transformacja danych makroekonomicznych...")
            transformed_macro = transform_macro_data(macro_data)

            for company in companies:
                company_id = company.id
                ticker = company.ticker
                sector_id = company.sector_id

                logger.info(f"Przetwarzanie spółki {ticker} (company_id: {company_id})...")
                try:
                    market_data = load_market_data(company_id)
                except Exception as e:
                    logger.error(f"Błąd podczas pobierania danych rynkowych dla {ticker}: {e}")
                    continue

                if market_data.empty:
                    logger.info(f"Brak danych rynkowych dla {ticker}, pomijanie...")
                    continue

                fundamentals_data = load_fundamentals(company_id)
                transformed_fundamentals = transform_fundamentals(fundamentals_data)

                if market_data.empty or transformed_fundamentals.empty:
                    logger.info(f"Brak danych do zapisania dla {ticker}")
                    continue

                market_data['date'] = pd.to_datetime(market_data['date'])
                transformed_fundamentals['date'] = pd.to_datetime(transformed_fundamentals['date'])

                min_date = max(market_data['date'].min(), transformed_fundamentals['date'].min(), transformed_macro['date'].min())
                max_date = min(market_data['date'].max(), transformed_fundamentals['date'].max(), transformed_macro['date'].max())

                merged = pd.merge(
                    transformed_fundamentals,
                    market_data,
                    on=["company_id", "date"],  
                    how="inner"
                )
                if merged.empty:
                    logger.info(f"Brak wspólnych dat dla danych rynkowych i fundamentalnych dla {ticker}")
                    continue

                transformed_macro['date'] = pd.to_datetime(transformed_macro['date'])

                final_df = pd.merge(
                    merged,
                    transformed_macro,
                    on="date",
                    how="left"
                )
                if final_df.empty:
                    logger.info(f"Brak danych do zapisania po dodaniu makroekonomicznych dla {ticker}")
                    continue

                final_df = final_df[final_df['date'] >= min_date]
                final_df = final_df[final_df['date'] <= max_date]

                try:
                    final_df = final_df.sort_values("date").drop_duplicates(subset=["company_id", "date"], keep="last")
                    final_df["sector_id"] = sector_id

                    for _, row in final_df.iterrows():
                        stmt = insert(FeaturesPreparedWig20).values(**row.to_dict())
                        stmt = stmt.on_conflict_do_nothing(index_elements=["company_id", "date"])
                        self.db.execute(stmt)
                    
                    self.db.commit()
                    logger.info(f"Zapisano {len(final_df)} rekordów dla {ticker}")
                except Exception as e:
                    logger.error(f"Błąd podczas zapisywania danych dla {ticker}: {e}")
                    self.db.rollback()
        except Exception as e:
            logger.error(f"Błąd w pipeline: {e}")
        finally:
            self.db.close()

if __name__ == "__main__":
    pipeline = FundamentalsBranchPipeline()
    pipeline.run()
