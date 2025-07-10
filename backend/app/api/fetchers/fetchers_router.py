def include_fetcher_routers(app):
    from app.api.fetchers import (
        ohlcv_fetcher,
        features_fetcher,
        fundamentals_fetcher,
        macro_fetcher,
        news_articles_fetcher
    )

    app.include_router(ohlcv_fetcher.router)
    app.include_router(features_fetcher.router)
    app.include_router(fundamentals_fetcher.router)
    app.include_router(macro_fetcher.router)
    app.include_router(news_articles_fetcher.router)