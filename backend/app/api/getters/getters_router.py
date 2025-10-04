def include_getter_routers(app):
    from app.api.getters import (
        ohlcv_getter,
        features_getter,
        fundamentals_getter,
        macro_getter,
        news_articles_getter,
        companies_getter,
        gpw_calendar_getter,
    )

    app.include_router(ohlcv_getter.router)
    app.include_router(features_getter.router)
    app.include_router(fundamentals_getter.router)
    app.include_router(macro_getter.router)
    app.include_router(news_articles_getter.router)
    app.include_router(companies_getter.router)
    app.include_router(gpw_calendar_getter.router)