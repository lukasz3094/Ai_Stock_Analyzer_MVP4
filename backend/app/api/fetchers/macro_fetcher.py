from fastapi import APIRouter
from app.services.fetchers.macro_fetcher import update_macro_data_table

router = APIRouter()

@router.post("/macro/update")
def update_macro():
    update_macro_data_table()
    return {"status": "Dane makro zostały zaktualizowane."}
