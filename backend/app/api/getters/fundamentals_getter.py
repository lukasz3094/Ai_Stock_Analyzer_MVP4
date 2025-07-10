from fastapi import APIRouter
from app.services.getters.fundamentals_getter import (get_all_fundamentals, get_fundamentals_by_company, get_fundamentals_by_date, 
                                                         get_fundamentals_by_company_and_date, get_fundamentals_from_date, get_fundamentals_by_company_from_date)

router = APIRouter()


@router.get("/fundamentals/all")
def all_fundamentals():
    fundamentals = get_all_fundamentals()
    return {"fundamentals": fundamentals}

@router.get("/fundamentals/company/{company_id}")
def fundamentals_by_company(company_id: int):
    fundamentals = get_fundamentals_by_company(company_id)
    return {"fundamentals": fundamentals}

@router.get("/fundamentals/date/{date}")
def fundamentals_by_date(date: str):
    fundamentals = get_fundamentals_by_date(date)
    return {"fundamentals": fundamentals}

@router.get("/fundamentals/company/{company_id}/date/{date}")
def fundamentals_by_company_and_date(company_id: int, date: str):
    fundamentals = get_fundamentals_by_company_and_date(company_id, date)
    return {"fundamentals": fundamentals}

@router.get("/fundamentals/from_date/{from_date}")
def fundamentals_from_date(from_date: str):
    fundamentals = get_fundamentals_from_date(from_date)
    return {"fundamentals": fundamentals}

@router.get("/fundamentals/company/{company_id}/from_date/{from_date}")
def fundamentals_by_company_from_date(company_id: int, from_date: str):
    fundamentals = get_fundamentals_by_company_from_date(company_id, from_date)
    return {"fundamentals": fundamentals}