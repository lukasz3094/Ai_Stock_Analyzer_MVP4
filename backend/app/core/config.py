import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

engine = create_engine(DB_URL,
    pool_size=20,
    max_overflow=20,
    pool_timeout=30
)

SessionLocal = scoped_session(sessionmaker(bind=engine))
Base = declarative_base()
