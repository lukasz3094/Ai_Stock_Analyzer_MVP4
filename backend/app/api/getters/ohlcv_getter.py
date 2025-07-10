from fastapi import APIRouter
from app.services.getters.ohlcv_getter import (get_all_ohlcv, get_ohlcv_by_company, get_ohlcv_by_date,
                                                  get_ohlcv_by_company_and_date, get_ohlcv_from_date, get_ohlcv_by_company_and_from_date)

router = APIRouter()


@router.get("/ohlcv/all")
def all_ohlcv():
    ohlcv = get_all_ohlcv()
    return {"ohlcv": ohlcv}

@router.get("/ohlcv/company/{company_id}")
def ohlcv_by_company(company_id: int):
    ohlcv = get_ohlcv_by_company(company_id)
    return {"ohlcv": ohlcv}

@router.get("/ohlcv/date/{date}")
def ohlcv_by_date(date: str):
    ohlcv = get_ohlcv_by_date(date)
    return {"ohlcv": ohlcv}

@router.get("/ohlcv/company/{company_id}/date/{date}")
def ohlcv_by_company_and_date(company_id: int, date: str):
    ohlcv = get_ohlcv_by_company_and_date(company_id, date)
    return {"ohlcv": ohlcv}

@router.get("/ohlcv/from_date/{from_date}")
def ohlcv_from_date(from_date: str):
    ohlcv = get_ohlcv_from_date(from_date)
    return {"ohlcv": ohlcv}

@router.get("/ohlcv/company/{company_id}/from_date/{from_date}")
def ohlcv_by_company_from_date(company_id: int, from_date: str):
    ohlcv = get_ohlcv_by_company_and_from_date(company_id, from_date)
    return {"ohlcv": ohlcv}