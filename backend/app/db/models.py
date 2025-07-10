from sqlalchemy import Column, String, Integer, Numeric, Date, ForeignKey
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
    company_id = Column(Integer, ForeignKey("companies.id"))
    date = Column(Date, nullable=False)
    headline = Column(String, nullable=False)
    content = Column(String, nullable=False)
    url = Column(String, nullable=False)
    source = Column(String, nullable=False)