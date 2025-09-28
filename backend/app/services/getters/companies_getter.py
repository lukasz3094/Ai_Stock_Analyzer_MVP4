from app.core import logger
from app.core.config import SessionLocal
from app.db.models import Company, CompanyAlias


def get_all_companies():
    db = SessionLocal()
    try:
        companies = db.query(Company).all()
        return companies
    except Exception as e:
        logger.error(f"Error fetching companies: {e}")
        return []
    finally:
        db.close()
    
def get_company_by_id(company_id):
    db = SessionLocal()
    try:
        company = db.query(Company).filter(Company.id == company_id).first()
        return company
    except Exception as e:
        logger.error(f"Error fetching company with ID {company_id}: {e}")
        return None
    finally:
        db.close()

def get_company_by_ticker(ticker):
    db = SessionLocal()
    try:
        company = db.query(Company).filter(Company.ticker == ticker).first()
        return company
    except Exception as e:
        logger.error(f"Error fetching companies with ticker {ticker}: {e}")
        return None
    finally:
        db.close()

def get_companies_by_sector_id(sector_id):
    db = SessionLocal()
    try:
        companies = db.query(Company).filter(Company.sector_id == sector_id).all()
        return companies
    except Exception as e:
        logger.error(f"Error fetching companies for sector ID {sector_id}: {e}")
        return []
    finally:
        db.close()

def get_all_company_aliases():
    db = SessionLocal()
    try:
        aliases = db.query(CompanyAlias).all()
        return aliases
    except Exception as e:
        logger.error(f"Error fetching company aliases: {e}")
        return []
    finally:
        db.close()

def get_company_alias_by_id(alias_id):
    db = SessionLocal()
    try:
        alias = db.query(CompanyAlias).filter(CompanyAlias.id == alias_id).first()
        return alias
    except Exception as e:
        logger.error(f"Error fetching company alias with ID {alias_id}: {e}")
        return None
    finally:
        db.close()

def get_company_aliases_by_company_id(company_id):
    db = SessionLocal()
    try:
        aliases = db.query(CompanyAlias).filter(CompanyAlias.company_id == company_id).all()
        return aliases
    except Exception as e:
        logger.error(f"Error fetching company aliases for company ID {company_id}: {e}")
        return []
    finally:
        db.close()

def get_companies_by_group(group_name):
    db = SessionLocal()
    try:
        companies = db.query(Company).filter(Company.groups.any(group_name)).all()
        return companies
    except Exception as e:
        logger.error(f"Error fetching companies for group {group_name}: {e}")
        return []
    finally:
        db.close()
