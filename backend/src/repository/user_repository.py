from typing import Optional

from sqlalchemy import select

from backend.src.constants import States, Roles
from backend.src.core import async_db_session
from backend.src.core import verify_password
from backend.src.dto import (UserCreateDTO,
                             UserUpdateDTO)
from backend.src.models import UserEntity
from backend.src.repository import BaseRepository


class UserRepository(BaseRepository[UserEntity, UserCreateDTO, UserUpdateDTO]):

    async def get_by_username(self, username: str) -> Optional[UserEntity]:
        async with async_db_session() as session:
            query = select(self.model)
            query = query.where(UserEntity.username == username)
            query = self.with_nolock(query)
            query = await session.execute(query)
            query = query.scalars().first()
            return query

    async def authenticate(self, username: str, password: str) -> Optional[UserEntity]:
        user = await self.get_by_username(username=username)
        if not user or user.password is None:
            return None
        if not verify_password(password, user.password):
            return None
        return user

    def is_active(self, user: UserEntity) -> bool:
        return user.state == States.Active.value

    def is_superuser(self, user: UserEntity) -> bool:
        return user.role == Roles.Admin.value or user.role == Roles.System.value

    def is_admin(self, user: UserEntity) -> bool:
        return user.role == Roles.Admin.value


user_repository = UserRepository(UserEntity)
