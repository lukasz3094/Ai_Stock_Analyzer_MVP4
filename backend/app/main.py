from fastapi import FastAPI
from app.api.fetchers.fetchers_router import include_fetcher_routers
from app.api.getters.getters_router import include_getter_routers


app = FastAPI(title="Stock Market Analysis API", version="0.1")
include_fetcher_routers(app)
include_getter_routers(app)
