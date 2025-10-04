from datetime import datetime
from app.db.enums import SentimentLabel
from sqlalchemy import Column, String, Integer, Numeric, Date, ForeignKey, Boolean, DateTime, Enum, ARRAY
from sqlalchemy.dialects.postgresql import JSONB
from app.core.config import Base

class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), unique=True, nullable=False)
    name = Column(String, nullable=False)
    ticker_bankier = Column(String(10), nullable=True)
    sector_id = Column(Integer, ForeignKey("sectors.id"))
    is_active = Column(Boolean, nullable=False)
    groups = Column(ARRAY(String), nullable=True)

class Sector(Base):
    __tablename__ = "sectors"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    category = Column(String, nullable=False)

class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"))
    date = Column(Date, nullable=False)
    open = Column(Numeric)
    high = Column(Numeric)
    low = Column(Numeric)
    close = Column(Numeric)
    volume = Column(Integer)

class MarketFeatures(Base):
    __tablename__ = "market_features"

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"))
    date = Column(Date, nullable=False)
    rsi = Column(Numeric)
    macd = Column(Numeric)
    sma = Column(Numeric)
    ema = Column(Numeric)
    bollinger_upper = Column(Numeric)
    bollinger_lower = Column(Numeric)
    momentum = Column(Numeric)
    stochastic = Column(Numeric)

class Fundamentals(Base):
    __tablename__ = "fundamentals"

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"))
    date = Column(Date, nullable=False)
    revenue = Column(Numeric)
    operating_profit = Column(Numeric)
    gross_profit = Column(Numeric)
    net_profit = Column(Numeric)
    ebitda = Column(Numeric)

class MacroData(Base):
    __tablename__ = "macro_data"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, unique=True)
    gdp = Column(Numeric, nullable=True)
    cpi = Column(Numeric, nullable=True)
    unemployment_rate = Column(Numeric, nullable=True)
    interest_rate = Column(Numeric, nullable=True)
    exchange_rate_eur = Column(Numeric, nullable=True)
    exchange_rate_usd = Column(Numeric, nullable=True)

class NewsArticles(Base):
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    headline = Column(String, nullable=False)
    content = Column(String, nullable=False)
    url = Column(String, nullable=False)
    source = Column(String, nullable=False)

class CompanyAlias(Base):
    __tablename__ = "company_aliases"

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"))
    alias = Column(String, nullable=False)
    is_primary = Column(Boolean, nullable=False)
    source = Column(String, nullable=False)

class GpwSessions(Base):
    __tablename__ = "gpw_sessions"

    trade_date = Column(Date, primary_key=True)
    is_trading_day = Column(Boolean, nullable=False)

class ContextTag(Base):
    __tablename__ = "context_tags"

    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False)
    name = Column(String, unique=True, nullable=False)

class CompanyContextTagsList(Base):
    __tablename__ = "company_context_tags_list"

    company_id = Column(Integer, ForeignKey("companies.id", ondelete="CASCADE"), primary_key=True)
    context_tag_ids = Column(ARRAY(Integer), nullable=False)

class NewsFeaturesPrepared(Base):
    __tablename__ = "news_features_prepared"

    id = Column(Integer, primary_key=True)
    news_article_id = Column(Integer, ForeignKey("news_articles.id"))
    company_id = Column(Integer, ForeignKey("companies.id"))
    sector_id = Column(Integer, ForeignKey("sectors.id"))
    context_tag_id = Column(Integer, ForeignKey("context_tags.id"))
    confidence_score = Column(Numeric, nullable=False)
    sentiment_score = Column(Numeric, nullable=True)
    sentiment_label = Column(Enum(SentimentLabel), nullable=True)

class MacroFeaturesPrepared(Base):
    __tablename__ = "macro_features_prepared"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, nullable=False)
    gdp = Column(Numeric, nullable=True)
    cpi = Column(Numeric, nullable=True)
    unemployment_rate = Column(Numeric, nullable=True)
    interest_rate = Column(Numeric, nullable=True)
    exchange_rate_eur = Column(Numeric, nullable=True)
    exchange_rate_usd = Column(Numeric, nullable=True)

class FundamentalsFeaturesPrepared(Base):
    __tablename__ = "fundamentals_features_prepared"

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    date = Column(Date, nullable=False)
    revenue = Column(Numeric, nullable=True)
    operating_profit = Column(Numeric, nullable=True)
    gross_profit = Column(Numeric, nullable=True)
    net_profit = Column(Numeric, nullable=True)
    ebitda = Column(Numeric, nullable=True)

class MarketFeaturesPrepared(Base):
    __tablename__ = "market_features_prepared"

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Numeric, nullable=True)
    high = Column(Numeric, nullable=True)
    low = Column(Numeric, nullable=True)
    close = Column(Numeric, nullable=True)
    volume = Column(Integer, nullable=True)
    sma_14 = Column(Numeric, nullable=True)
    ema_14 = Column(Numeric, nullable=True)
    rsi_14 = Column(Numeric, nullable=True)
    macd = Column(Numeric, nullable=True)
    macd_signal = Column(Numeric, nullable=True)
    macd_hist = Column(Numeric, nullable=True)

class SelectedFeatures(Base):
    __tablename__ = "selected_features"

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True)
    sector_id = Column(Integer, ForeignKey("sectors.id"), nullable=False)
    run_date = Column(Date, nullable=False)
    model_type = Column(String, nullable=False)
    selected_features = Column(JSONB, nullable=False)
    total_features = Column(Integer, nullable=False)
    feature_importances = Column(JSONB, nullable=True)
    notes = Column(String, nullable=True)

class FeaturesPreparedWig20(Base):
    __tablename__ = "features_prepared_wig20"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    sector_id = Column(Integer, ForeignKey("sectors.id"), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Numeric, nullable=True)
    high = Column(Numeric, nullable=True)
    low = Column(Numeric, nullable=True)
    close = Column(Numeric, nullable=True)
    volume = Column(Numeric, nullable=True)
    revenue = Column(Numeric, nullable=True)
    operating_profit = Column(Numeric, nullable=True)
    gross_profit = Column(Numeric, nullable=True)
    net_profit = Column(Numeric, nullable=True)
    ebitda = Column(Numeric, nullable=True)
    gdp = Column(Numeric, nullable=True)
    cpi = Column(Numeric, nullable=True)
    unemployment_rate = Column(Numeric, nullable=True)
    interest_rate = Column(Numeric, nullable=True)
    exchange_rate_eur = Column(Numeric, nullable=True)
    exchange_rate_usd = Column(Numeric, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

class FeaturesFinalPrepared(Base):
    __tablename__ = "features_final_prepared"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Numeric, nullable=True)
    high = Column(Numeric, nullable=True)
    low = Column(Numeric, nullable=True)
    close = Column(Numeric, nullable=True)
    volume = Column(Numeric, nullable=True)
    sma_14 = Column(Numeric, nullable=True)
    ema_14 = Column(Numeric, nullable=True)
    rsi_14 = Column(Numeric, nullable=True)
    macd = Column(Numeric, nullable=True)
    macd_signal = Column(Numeric, nullable=True)
    macd_hist = Column(Numeric, nullable=True)
    revenue = Column(Numeric, nullable=True)
    operating_profit = Column(Numeric, nullable=True)
    gross_profit = Column(Numeric, nullable=True)
    net_profit = Column(Numeric, nullable=True)
    ebitda = Column(Numeric, nullable=True)
    gdp = Column(Numeric, nullable=True)
    cpi = Column(Numeric, nullable=True)
    unemployment_rate = Column(Numeric, nullable=True)
    interest_rate = Column(Numeric, nullable=True)
    exchange_rate_eur = Column(Numeric, nullable=True)
    exchange_rate_usd = Column(Numeric, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    # confidence_score_avg = Column(Numeric, nullable=True)
    # confidence_score_sum = Column(Numeric, nullable=True)
    # news_count = Column(Integer, default=0)

class FeaturesFinalPreparedV2(Base):
    __tablename__ = "features_final_prepared_v2"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Numeric, nullable=True)
    high = Column(Numeric, nullable=True)
    low = Column(Numeric, nullable=True)
    close = Column(Numeric, nullable=True)
    volume = Column(Numeric, nullable=True)
    revenue = Column(Numeric, nullable=True)
    operating_profit = Column(Numeric, nullable=True)
    gross_profit = Column(Numeric, nullable=True)
    net_profit = Column(Numeric, nullable=True)
    ebitda = Column(Numeric, nullable=True)
    gdp = Column(Numeric, nullable=True)
    cpi = Column(Numeric, nullable=True)
    unemployment_rate = Column(Numeric, nullable=True)
    interest_rate = Column(Numeric, nullable=True)
    exchange_rate_eur = Column(Numeric, nullable=True)
    exchange_rate_usd = Column(Numeric, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

class FeaturesFinalPreparedV3(Base):
    __tablename__ = "features_final_prepared_v3"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    date = Column(Date, nullable=False)

    close = Column(Numeric, nullable=True)
    volume = Column(Numeric, nullable=True)

    ret_1d = Column(Numeric, nullable=True)
    ret_5d = Column(Numeric, nullable=True)
    ret_20d = Column(Numeric, nullable=True)
    vol_10d = Column(Numeric, nullable=True)
    vol_20d = Column(Numeric, nullable=True)

    px_sma14_dist = Column(Numeric, nullable=True)
    px_ema14_dist = Column(Numeric, nullable=True)
    rsi_c = Column(Numeric, nullable=True)
    macd_hist_norm = Column(Numeric, nullable=True)
    macd_minus_signal = Column(Numeric, nullable=True)
    vol_pressure_20d = Column(Numeric, nullable=True)

    revenue_yoy = Column(Numeric, nullable=True)
    ebitda_yoy = Column(Numeric, nullable=True)
    net_profit_yoy = Column(Numeric, nullable=True)
    gross_profit_yoy = Column(Numeric, nullable=True)

    gdp_yoy = Column(Numeric, nullable=True)
    cpi_yoy = Column(Numeric, nullable=True)
    rate_change = Column(Numeric, nullable=True)
    fx_eur_20d = Column(Numeric, nullable=True)
    fx_usd_20d = Column(Numeric, nullable=True)

    created_at = Column(DateTime, default=datetime.now)
