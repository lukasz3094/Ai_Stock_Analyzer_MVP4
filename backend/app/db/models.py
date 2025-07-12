from sqlalchemy import Column, String, Integer, Numeric, Date, ForeignKey, Boolean
from app.core.config import Base

class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), unique=True, nullable=False)
    ticker_alt = Column(String(10), nullable=True)
    name = Column(String, nullable=False)
    ticker_bankier = Column(String(10), nullable=True)
    sector_id = Column(Integer, ForeignKey("sectors.id"))

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

class ContextTag(Base):
    __tablename__ = "context_tags"

    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False)
    name = Column(String, unique=True, nullable=False)

class NewsFeaturesPrepared(Base):
    __tablename__ = "news_features_prepared"

    id = Column(Integer, primary_key=True)
    news_article_id = Column(Integer, ForeignKey("news_articles.id"))
    company_id = Column(Integer, ForeignKey("companies.id"))
    sector_id = Column(Integer, ForeignKey("sectors.id"))
    context_tag_id = Column(Integer, ForeignKey("context_tags.id"))
    confidence_score = Column(Numeric, nullable=False)

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
