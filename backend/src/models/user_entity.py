from typing import Optional

from sqlalchemy import (Integer)
from sqlalchemy import (String)
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

from backend.src.models import BaseEntity


class UserEntity(BaseEntity):
    __tablename__ = "user"

    username: Mapped[Optional[str]] = mapped_column(String(120), nullable=False, unique=True)
    password: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    role: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    state: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
