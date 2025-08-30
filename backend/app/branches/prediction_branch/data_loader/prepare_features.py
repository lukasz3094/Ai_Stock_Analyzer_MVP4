from datetime import timedelta
import pandas as pd
from tqdm import tqdm
from sqlalchemy import func, and_
from app.core.logger import logger
from app.core.config import SessionLocal
from app.db.models import (
    Company, MacroFeaturesPrepared, FundamentalsFeaturesPrepared,
    MarketFeaturesPrepared, NewsFeaturesPrepared, NewsArticles
)

def prepare_features(sector_id: int, end_date: str, mode: str = "auto"):
    end_date = pd.to_datetime(end_date)
    db = SessionLocal()

    try:
        company_ids = [row[0] for row in db.query(Company.id).filter(Company.sector_id == sector_id).all()]
        if not company_ids:
            logger.warning(f"No companies found for sector {sector_id}")
            return

        # Determine start_date
        if mode == "auto":
            last_date = db.query(func.min(MarketFeaturesPrepared.date))\
                        .filter(MarketFeaturesPrepared.company_id.in_(company_ids))\
                        .filter(MarketFeaturesPrepared.date <= end_date).scalar()
            if last_date:
                start_date = last_date + timedelta(days=1)
            else:
                mode = "min"

        if mode == "min":
            min_dates = []

            market_min = db.query(func.min(MarketFeaturesPrepared.date))\
                .filter(MarketFeaturesPrepared.company_id.in_(company_ids),
                        MarketFeaturesPrepared.date <= end_date).scalar()
            if market_min:
                min_dates.append(market_min)

            fund_min = db.query(func.min(FundamentalsFeaturesPrepared.date))\
                .filter(FundamentalsFeaturesPrepared.company_id.in_(company_ids),
                        FundamentalsFeaturesPrepared.date <= end_date).scalar()
            if fund_min:
                min_dates.append(fund_min)

            macro_min = db.query(func.min(MacroFeaturesPrepared.date))\
                .filter(MacroFeaturesPrepared.date <= end_date).scalar()
            if macro_min:
                min_dates.append(macro_min)

            # news_min = db.query(func.min(NewsArticles.date))\
            #     .join(NewsFeaturesPrepared, NewsArticles.id == NewsFeaturesPrepared.news_article_id)\
            #     .filter(NewsFeaturesPrepared.company_id.in_(company_ids),
            #             NewsArticles.date <= end_date).scalar()
            # if news_min:
            #     min_dates.append(news_min)

            start_date = max(min_dates) if min_dates else None

        start_date = pd.to_datetime(start_date)

        if not start_date:
            logger.warning(f"No valid start date found for sector {sector_id}")
            return

        if start_date > end_date:
            logger.warning(f"Start date {start_date.date()} is after end date {end_date.date()} for sector {sector_id}. Skipping.")
            return


        logger.info(f"Preparing features for sector {sector_id} from {start_date} to {end_date}")
        dates = pd.date_range(start=start_date, end=end_date, freq="B")

        all_rows = []
        for company_id in tqdm(company_ids, desc=f"Sector {sector_id}"):
            market = pd.read_sql_query(
                f"SELECT * FROM market_features_prepared WHERE company_id = {company_id}",
                db.bind).drop(columns="id", errors="ignore")

            fundamentals = pd.read_sql_query(
                f"SELECT * FROM fundamentals_features_prepared WHERE company_id = {company_id}",
                db.bind).drop(columns="id", errors="ignore")

            macro = pd.read_sql_query(
                "SELECT * FROM macro_features_prepared", db.bind
            ).drop(columns="id", errors="ignore")

            # news = pd.read_sql_query(f"""
            #     SELECT a.date,
            #            AVG(f.confidence_score) as confidence_score_avg,
            #            SUM(f.confidence_score) as confidence_score_sum,
            #            COUNT(*) as news_count
            #     FROM news_features_prepared f
            #     JOIN news_articles a ON f.news_article_id = a.id
            #     WHERE f.company_id = {company_id}
            #       AND f.confidence_score >= 0.7
            #     GROUP BY a.date
            # """, db.bind)

            df = pd.DataFrame({"date": dates})
            df["company_id"] = company_id

            market["date"] = pd.to_datetime(market["date"])
            fundamentals["date"] = pd.to_datetime(fundamentals["date"])
            macro["date"] = pd.to_datetime(macro["date"])
            # news["date"] = pd.to_datetime(news["date"])

            df = df.merge(market, on=["company_id", "date"], how="left")
            df = pd.merge_asof(df.sort_values("date"), fundamentals.sort_values("date"),
                               by="company_id", on="date", direction="backward")
            df = pd.merge_asof(df.sort_values("date"), macro.sort_values("date"),
                               on="date", direction="backward")
            # df = df.merge(news, on="date", how="left")

            all_rows.append(df)

        final_df = pd.concat(all_rows)
        final_df = final_df.dropna(subset=["open"])

        final_df.to_sql("features_final_prepared", db.bind,
                        if_exists="append", index=False, method="multi", chunksize=1000)

        logger.info(f"Inserted {len(final_df)} rows for sector {sector_id} from {start_date} to {end_date}")
    except Exception as e:
        logger.error(f"Error preparing features for sector {sector_id}: {e}")
        print(f"Error preparing features for sector {sector_id}: {e}")
    finally:
        db.close()

def prepare_features_for_all_sectors(end_date: str, mode: str = "auto"):
    db = SessionLocal()
    sectors = [row[0] for row in db.query(Company.sector_id).distinct().all()]
    db.close()

    for sector_id in sectors:
        prepare_features(sector_id, end_date, mode)

if __name__ == "__main__":
    prepare_features_for_all_sectors(end_date="2024-12-31", mode="auto")
