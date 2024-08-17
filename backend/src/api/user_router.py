from typing import List

from fastapi import APIRouter, Depends, HTTPException
from starlette import status

from backend.src.core import log_factory, get_password_hash
from backend.src.core import security
from backend.src.core.deps import get_current_active_user
from backend.src.dto import (LoginDTO, JWTTokenDTO, UserCreateDTO, UserUpdateDTO, UserDTO)
from backend.src.models import UserEntity
from backend.src.repository import user_repository

router = APIRouter()

logger = log_factory.get_logger(__name__)


@router.post("/auth/login/jwt", response_model=JWTTokenDTO)
async def login_access_token(
        *,
        form_data: LoginDTO
):
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    user = await user_repository.authenticate(
        username=form_data.username,
        password=form_data.password
    )
    if not user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect email or password")
    elif not user_repository.is_active(user):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")

    token, expire = security.create_access_token(
        dict(
            username=user.username,
            role=user.role
        ),
    )

    return JWTTokenDTO(
        access_token=token,
        token_type="bearer",
        expire=expire
    )


@router.get("/user", response_model=List[UserDTO])
async def read_users(
        skip: int = 0,
        limit: int = 100,
        current_user: UserEntity = Depends(get_current_active_user),
):
    res = await user_repository.get_multi(skip=skip, limit=limit)
    return [UserDTO(**i.serialize()) for i in res]


@router.post("/user", response_model=UserDTO)
async def create_user(
        *,
        user_in: UserCreateDTO,
        current_user: UserEntity = Depends(get_current_active_user),
):
    """
        Create new user.
        """
    hashed_password = get_password_hash(user_in.password)
    user_in.password = hashed_password
    user = await user_repository.get_by_username(user_in.username)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The user with this username already exists in the system.",
        )
    user = await user_repository.create(user_in)
    return UserDTO(**user.serialize())


@router.put("/user/{_id}", response_model=UserDTO)
async def update_user(
        *,
        _id: int,
        instance_in: UserUpdateDTO,
        current_user: UserEntity = Depends(get_current_active_user),
):
    if not user_repository.is_admin(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="user not found",
        )
    user = await user_repository.get(_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="user not found",
        )
    hashed_password = get_password_hash(instance_in.password)
    instance_in.password = hashed_password
    user = await user_repository.update(db_obj=user, obj_in=instance_in)
    return UserDTO(**user.serialize())


@router.get("/auth/current-user", response_model=UserDTO)
async def read_current_user(
        *,
        current_user: UserEntity = Depends(get_current_active_user),
):
    return UserDTO(**current_user.serialize())
