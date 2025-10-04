from fastapi import APIRouter
from app.services.getters.gpw_calendar_getter import get_all_gpwsessions, get_gpwsessions_by_range, get_gpwsessions_by_start_date, get_gpwsessions_by_end_date

router = APIRouter()

@router.get("/gpw_sessions/all")
def all_gpwsessions():
    gpw_sessions = get_all_gpwsessions()
    return {"gpw_sessions": gpw_sessions}

@router.get("/gpw_sessions/range")
def gpw_sessions_by_range(start_date: str, end_date: str):
    gpw_sessions = get_gpwsessions_by_range(start_date, end_date)
    return {"gpw_sessions": gpw_sessions}

@router.get("/gpw_sessions/start_date/{date}")
def gpw_sessions_by_start_date(date: str):
    gpw_sessions = get_gpwsessions_by_start_date(date)
    return {"gpw_sessions": gpw_sessions}

@router.get("/gpw_sessions/end_date/{date}")
def gpw_sessions_by_end_date(date: str):
    gpw_sessions = get_gpwsessions_by_end_date(date)
    return {"gpw_sessions": gpw_sessions}