def include_operations_routers(app):
    from app.api.operations import (
        news_classifier
    )

    app.include_router(news_classifier.router)