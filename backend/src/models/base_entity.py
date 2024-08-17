from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy import (BigInteger, DateTime, func)
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

from backend.src.core.global_log import structlog

logger = structlog.get_logger(__name__)


class Serializer(object):

    def serialize(self):
        serialized = {}
        for c in inspect(self).attrs.keys():
            serialized[c] = getattr(self, c)
        return serialized

    @staticmethod
    def serialize_list(rows: List[Any]):
        return [r.serialize() for r in rows]


class BaseEntity(AsyncAttrs, DeclarativeBase, Serializer):
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True, autoincrement=True)
    __name__: str
    __table_args__ = {"extend_existing": True}

    date_created: Mapped[Optional[datetime]] = mapped_column(DateTime, default=func.current_timestamp(),
                                                             nullable=True)

    date_modified: Mapped[Optional[datetime]] = mapped_column(DateTime, default=func.current_timestamp(),
                                                              onupdate=func.current_timestamp(), nullable=True, )

    date_archived: Mapped[Optional[Any]] = mapped_column(DateTime, default=None, nullable=True)

    __mapper_args__ = {"eager_defaults": True}

    # Generate __tablename__ automatically
    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()
