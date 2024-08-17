from fastapi import APIRouter

from backend.src.api import data_source_router
from backend.src.api import user_router
from backend.src.core.global_log import log_factory

logger = log_factory.get_logger(__name__)

api_router = APIRouter()

api_router.include_router(user_router.router, tags=["auth", "user"])
api_router.include_router(data_source_router.router, prefix="/data-source", tags=["data-source"])
