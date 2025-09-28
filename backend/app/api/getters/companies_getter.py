from fastapi import APIRouter
from app.services.getters.companies_getter import get_companies_by_group

router = APIRouter()


@router.get("/companies/group/{group_name}")
def companies_by_group(group_name: str):
    companies = get_companies_by_group(group_name)
    return {"companies": companies}
