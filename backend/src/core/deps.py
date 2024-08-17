import json
from typing import Optional, Annotated

from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer
from jose import jwt
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from backend.src.core import async_db_session, settings, ALGORITHM, log_factory
from backend.src.dto import JWTTokenPayloadDTO
from backend.src.models import UserEntity
from backend.src.repository import user_repository

logger = log_factory.get_logger(__name__)

oauth2_scheme = HTTPBearer()


async def get_db() -> AsyncSession:
    with async_db_session() as session:
        yield session
        session.close()


async def get_current_user(credentials: Annotated[str, Depends(oauth2_scheme)]) -> Optional[UserEntity]:
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        subject = json.loads(payload["sub"])
        token_data = JWTTokenPayloadDTO(**subject)
    except Exception as e:
        logger.error("get_current_user", error=e)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    user = await user_repository.get_by_username(token_data.username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


async def get_current_active_user(
        current_user: UserEntity = Depends(get_current_user),
) -> UserEntity:
    if not user_repository.is_active(current_user):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user


async def get_current_active_superuser(
        current_user: UserEntity = Depends(get_current_user),
) -> UserEntity:
    if not user_repository.is_superuser(current_user):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="The user doesn't have enough privileges"
        )
    return current_user
