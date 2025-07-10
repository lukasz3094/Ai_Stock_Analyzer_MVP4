from fastapi import APIRouter
from app.services.getters.features_getter import get_all_features, get_features_by_company, get_features_by_date, get_features_by_company_and_date, get_features_from_date

router = APIRouter()


@router.get("/features/all")
def all_features():
    features = get_all_features()
    return {"features": features}

@router.get("/features/company/{company_id}")
def features_by_company(company_id: int):
    features = get_features_by_company(company_id)
    return {"features": features}

@router.get("/features/date/{date}")
def features_by_date(date: str):
    features = get_features_by_date(date)
    return {"features": features}

@router.get("/features/company/{company_id}/date/{date}")
def features_by_company_and_date(company_id: int, date: str):
    features = get_features_by_company_and_date(company_id, date)
    return {"features": features}

def features_from_date(from_date: str):
    features = get_features_from_date(from_date)
    return {"features": features}
    