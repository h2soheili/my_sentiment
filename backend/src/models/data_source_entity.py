from typing import Any, Optional

from sqlalchemy import (Integer, BigInteger, DateTime)
from sqlalchemy import (String)
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

from backend.src.models import BaseEntity


class DataSourceEntity(BaseEntity):
    __tablename__ = 'data_source'

    key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=False)
    url: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=False)
    upto: Mapped[Optional[Any]] = mapped_column(DateTime, nullable=True, index=False)
    total_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    state: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
