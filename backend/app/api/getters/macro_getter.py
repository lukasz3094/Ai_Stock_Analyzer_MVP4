from fastapi import APIRouter
from app.services.getters.macro_getter import (get_all_macros, get_macro_by_date, get_macros_from_date)

router = APIRouter()


@router.get("/macros/all")
def all_macros():
    macros = get_all_macros()
    return {"macros": macros}

@router.get("/macros/date/{date}")
def macro_by_date(date: str):
    macros = get_macro_by_date(date)
    return {"macros": macros}

@router.get("/macros/from_date/{from_date}")
def macro_from_date(from_date: str):
    macros = get_macros_from_date(from_date)
    return {"macros": macros}
