from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from sqlalchemy import text
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from backend.src.api import api_router
from backend.src.core import log_factory, settings, async_db_session

logger = log_factory.get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print(_app)


app = FastAPI(
    title="project",
    version="1.0.0",
    description=" description",
    openapi_url=f"/openapi.json",
    debug=True,
    # lifespan=lifespan
)

app.include_router(api_router)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.mount("/static", StaticFiles(directory="static"), name="static")

swagger_js_url = "/static/swagger-ui-bundle.js"
swagger_css_url = "/static/swagger-ui.css"


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url=swagger_js_url,
        swagger_css_url=swagger_css_url,
    )


@app.get("/")
async def running():
    return f"Server is running  on {settings.SERVER_HOST}:{settings.SERVER_PORT}"


@app.on_event("startup")
async def startup_event():
    print("startup")
    async with async_db_session() as session:
        await session.execute(text("select 1"))


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=3000, reload=False, workers=1)

    # uvicorn.run('server_model:app', host="0.0.0.0", port=3000, reload=True, workers=1)
