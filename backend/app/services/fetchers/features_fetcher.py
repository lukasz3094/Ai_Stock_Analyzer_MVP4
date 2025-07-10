import pandas as pd
import pandas_ta as ta
from sqlalchemy.exc import SQLAlchemyError
from app.core.config import SessionLocal
from app.db.models import Company, MarketData, MarketFeatures
from app.core.logger import logger


def calculate_indicators(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.dropna().sort_values("date").copy()

    if len(df) < 20:
        logger.warning(f"[{ticker}]: zbyt mało sesji: {len(df)} (min. 20)")
        return pd.DataFrame()

    # Konwersja typów
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["rsi"] = ta.rsi(df["close"], length=14)
    df["macd"] = ta.macd(df["close"])["MACD_12_26_9"]
    df["sma"] = ta.sma(df["close"], length=14)
    df["ema"] = ta.ema(df["close"], length=14)

    bb = ta.bbands(df["close"], length=14)
    if bb is not None and "BBU_14_2.0" in bb:
        df["bollinger_upper"] = bb["BBU_14_2.0"]
        df["bollinger_lower"] = bb["BBL_14_2.0"]
    else:
        df["bollinger_upper"] = None
        df["bollinger_lower"] = None

    df["momentum"] = ta.mom(df["close"], length=10)
    df["stochastic"] = ta.stoch(df["high"], df["low"], df["close"])["STOCHk_14_3_3"]

    return df

def update_features_all_companies():
    db = SessionLocal()
    companies = db.query(Company).all()

    i = 1
    number_of_companies = len(companies)

    for company in companies:
        try:
            logger.info(f"({i} z {number_of_companies}) [{company.ticker}]: obliczanie wskaźników...")

            i += 1

            rows = (
                db.query(MarketData)
                .filter_by(company_id=company.id)
                .order_by(MarketData.date)
                .all()
            )
            if not rows:
                logger.warning(f"[{company.ticker}]: brak danych OHLCV.")
                continue

            df = pd.DataFrame([{
                "date": r.date,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume
            } for r in rows])

            df = calculate_indicators(df, company.ticker)

            for _, row in df.iterrows():
                if pd.isna(row["rsi"]):
                    continue  # ignorujemy początek

                feat = MarketFeatures(
                    company_id=company.id,
                    date=row["date"],
                    rsi=row["rsi"],
                    macd=row["macd"],
                    sma=row["sma"],
                    ema=row["ema"],
                    bollinger_upper=row["bollinger_upper"],
                    bollinger_lower=row["bollinger_lower"],
                    momentum=row["momentum"],
                    stochastic=row["stochastic"]
                )
                db.merge(feat)

            db.commit()
            logger.info(f"[{company.ticker}]: wskaźniki zapisane.")

        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"[{company.ticker}]: błąd SQL: {e}")
        except Exception as e:
            logger.error(f"[{company.ticker}]: wyjątek: {e}")

    db.close()
