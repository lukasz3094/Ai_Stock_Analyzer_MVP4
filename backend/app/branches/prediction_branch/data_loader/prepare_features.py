from datetime import timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm
from sqlalchemy import func, and_
from app.core.logger import logger
from app.core.config import SessionLocal
from app.db.models import (
    Company, MacroFeaturesPrepared, FundamentalsFeaturesPrepared, MarketData,
    MarketFeaturesPrepared, NewsFeaturesPrepared, NewsArticles
)

# With calculations (SME, RSI, etc.) v1

def prepare_features_with_calculations_v1(sector_id: int, end_date: str, mode: str = "auto"):
    end_date = pd.to_datetime(end_date)
    db = SessionLocal()

    try:
        company_ids = [row[0] for row in db.query(Company.id).filter(Company.sector_id == sector_id, Company.is_active == True).all()]
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

def prepare_features_for_all_sectors_with_calculations_v1(end_date: str, mode: str = "auto"):
    db = SessionLocal()
    sectors = [row[0] for row in db.query(Company.sector_id).distinct().all()]
    db.close()

    for sector_id in sectors:
        prepare_features_with_calculations_v1(sector_id, end_date, mode)

# No calculations (raw data only)

def prepare_features_without_calculations_v2(sector_id: int, end_date: str, mode: str = "auto"):
    end_date = pd.to_datetime(end_date)
    db = SessionLocal()

    try:
        company_ids = [row[0] for row in db.query(Company.id).filter(Company.sector_id == sector_id, Company.is_active == True).all()]
        if not company_ids:
            logger.warning(f"No companies found for sector {sector_id}")
            return

        # Determine start_date
        if mode == "auto":
            last_date = db.query(func.min(MarketData.date))\
                        .filter(MarketData.company_id.in_(company_ids))\
                        .filter(MarketData.date <= end_date).scalar()
            if last_date:
                start_date = last_date + timedelta(days=1)
            else:
                mode = "min"

        if mode == "min":
            min_dates = []

            market_min = db.query(func.min(MarketData.date))\
                .filter(MarketData.company_id.in_(company_ids),
                        MarketData.date <= end_date).scalar()
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
                f"SELECT * FROM market_data WHERE company_id = {company_id}",
                db.bind).drop(columns="id", errors="ignore")

            fundamentals = pd.read_sql_query(
                f"SELECT * FROM fundamentals_features_prepared WHERE company_id = {company_id}",
                db.bind).drop(columns="id", errors="ignore")

            macro = pd.read_sql_query(
                "SELECT * FROM macro_features_prepared", db.bind
            ).drop(columns="id", errors="ignore")

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

        final_df.to_sql("features_final_prepared_v2", db.bind,
                        if_exists="append", index=False, method="multi", chunksize=1000)

        logger.info(f"Inserted {len(final_df)} rows for sector {sector_id} from {start_date} to {end_date}")
    except Exception as e:
        logger.error(f"Error preparing features for sector {sector_id}: {e}")
        print(f"Error preparing features for sector {sector_id}: {e}")
    finally:
        db.close()

def prepare_features_for_all_sectors_without_calculations_v2(end_date: str, mode: str = "auto"):
    db = SessionLocal()
    sectors = [row[0] for row in db.query(Company.sector_id).distinct().all()]
    db.close()

    for sector_id in sectors:
        prepare_features_without_calculations_v2(sector_id, end_date, mode)

# Transformations (log returns, vol, etc.) v3
def prepare_features_transformed_v3(sector_id: int, end_date: str, mode: str = "auto"):
    end_date = pd.to_datetime(end_date)
    db = SessionLocal()
    try:
        company_ids = [r[0] for r in db.query(Company.id)
                       .filter(Company.sector_id == sector_id, Company.is_active == True).all()]
        if not company_ids:
            logger.warning(f"No companies for sector {sector_id}")
            return
        
        if mode == "auto":
            last_date = (db.query(func.min(MarketFeaturesPrepared.date))
                           .filter(MarketFeaturesPrepared.company_id.in_(company_ids))
                           .filter(MarketFeaturesPrepared.date <= end_date).scalar())
            if last_date:
                start_date = last_date + timedelta(days=1)
            else:
                mode = "min"

        if mode == "min":
            min_dates = []
            market_min = (db.query(func.min(MarketFeaturesPrepared.date))
                            .filter(MarketFeaturesPrepared.company_id.in_(company_ids),
                                    MarketFeaturesPrepared.date <= end_date).scalar())
            if market_min: min_dates.append(market_min)
            fund_min = (db.query(func.min(FundamentalsFeaturesPrepared.date))
                          .filter(FundamentalsFeaturesPrepared.company_id.in_(company_ids),
                                  FundamentalsFeaturesPrepared.date <= end_date).scalar())
            if fund_min: min_dates.append(fund_min)
            macro_min = (db.query(func.min(MacroFeaturesPrepared.date))
                           .filter(MacroFeaturesPrepared.date <= end_date).scalar())
            if macro_min: min_dates.append(macro_min)
            start_date = max(min_dates) if min_dates else None

        if not start_date or pd.to_datetime(start_date) > end_date:
            logger.warning(f"Invalid date range for sector {sector_id}")
            return

        dates = pd.date_range(start=pd.to_datetime(start_date), end=end_date, freq="B")
        all_rows = []

        for company_id in tqdm(company_ids, desc=f"Sector {sector_id} V3"):
            # v1 market table already has SMA/EMA/RSI/MACD — we’ll transform them
            market = pd.read_sql_query(
                f"SELECT company_id, date, open, high, low, close, volume, "
                f"sma_14, ema_14, rsi_14, macd, macd_signal, macd_hist "
                f"FROM market_features_prepared WHERE company_id = {company_id}",
                db.bind
            )

            fundamentals = pd.read_sql_query(
                f"SELECT * FROM fundamentals_features_prepared WHERE company_id = {company_id}",
                db.bind
            )

            macro = pd.read_sql_query(
                "SELECT * FROM macro_features_prepared", db.bind
            )

            # skeleton
            df = pd.DataFrame({"date": dates})
            df["company_id"] = company_id

            for t in (market, fundamentals, macro):
                t["date"] = pd.to_datetime(t["date"])

            # merge (same logic as V1)
            df = df.merge(market, on=["company_id", "date"], how="left")
            df = pd.merge_asof(df.sort_values("date"), fundamentals.sort_values("date"),
                               by="company_id", on="date", direction="backward")
            df = pd.merge_asof(df.sort_values("date"), macro.sort_values("date"),
                               on="date", direction="backward")

            df = df.dropna(subset=["close"]).sort_values(["company_id", "date"]).reset_index(drop=True)

            # ---- transforms ----
            # log returns
            log_close = np.log(df["close"].astype(float))
            df["ret_1d"] = log_close.diff(1)
            df["ret_5d"] = log_close.diff(5)
            df["ret_20d"] = log_close.diff(20)

            # realized vol
            r1 = df["ret_1d"]
            df["vol_10d"] = r1.rolling(10).std()
            df["vol_20d"] = r1.rolling(20).std()

            # distance to moving averages (scale-invariant)
            df["px_sma14_dist"] = df["close"] / df["sma_14"] - 1.0
            df["px_ema14_dist"] = df["close"] / df["ema_14"] - 1.0

            # RSI centered, MACD normalized
            df["rsi_c"] = (df["rsi_14"] - 50.0) / 50.0
            df["macd_hist_norm"] = df["macd_hist"] / df["ema_14"]
            df["macd_minus_signal"] = df["macd"] - df["macd_signal"]

            # volume pressure
            vol_mean20 = df["volume"].rolling(20).mean()
            df["vol_pressure_20d"] = df["volume"] / vol_mean20 - 1.0

            # fundamentals YoY (works if roughly quarterly; adjust 4→periods per year as needed)
            for col in ["revenue", "ebitda", "net_profit", "gross_profit"]:
                if col in df.columns:
                    df[f"{col}_yoy"] = df.groupby("company_id")[col].pct_change(4)

            # macro (YoY & changes)
            if "gdp" in df: df["gdp_yoy"] = df["gdp"].pct_change(4)
            if "cpi" in df: df["cpi_yoy"] = df["cpi"].pct_change(12)
            if "interest_rate" in df: df["rate_change"] = df["interest_rate"].diff(1)
            if "exchange_rate_eur" in df: df["fx_eur_20d"] = np.log(df["exchange_rate_eur"]).diff(20)
            if "exchange_rate_usd" in df: df["fx_usd_20d"] = np.log(df["exchange_rate_usd"]).diff(20)

            keep_cols = [
                "company_id","date","close","volume",
                "ret_1d","ret_5d","ret_20d","vol_10d","vol_20d",
                "px_sma14_dist","px_ema14_dist","rsi_c","macd_hist_norm","macd_minus_signal",
                "vol_pressure_20d",
                "revenue_yoy","ebitda_yoy","net_profit_yoy","gross_profit_yoy",
                "gdp_yoy","cpi_yoy","rate_change","fx_eur_20d","fx_usd_20d"
            ]
            df = df.reindex(columns=[c for c in keep_cols if c in df.columns])
            all_rows.append(df)

        final_df = pd.concat(all_rows, ignore_index=True)
        final_df = final_df.dropna(subset=["close"])  # allow NaNs in other cols; model can handle

        final_df.to_sql("features_final_prepared_v3", db.bind,
                        if_exists="append", index=False, method="multi", chunksize=1000)

        logger.info(f"V3 inserted {len(final_df)} rows for sector {sector_id}")
    except Exception as e:
        logger.error(f"V3 prepare error sector {sector_id}: {e}")
        print(f"Error preparing V3 features for sector {sector_id}: {e}")
    finally:
        db.close()

def prepare_features_for_all_sectors_with_transformations_v3(end_date: str, mode: str = "auto"):
    db = SessionLocal()
    sectors = [row[0] for row in db.query(Company.sector_id).distinct().all()]
    db.close()

    for sector_id in sectors:
        prepare_features_transformed_v3(sector_id, end_date, mode)

if __name__ == "__main__":
    # prepare_features_for_all_sectors_with_calculations_v1(end_date="2024-12-31", mode="auto")
    # prepare_features_for_all_sectors_without_calculations_v2(end_date="2024-12-31", mode="auto")
    prepare_features_for_all_sectors_with_transformations_v3(end_date="2024-12-31", mode="auto")
