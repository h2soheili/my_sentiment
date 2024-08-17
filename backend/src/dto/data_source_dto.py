from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class DataSourceCreateDTO(BaseModel):
    key: Optional[str] = None
    name: Optional[str] = None
    url: Optional[str] = None
    upto: Optional[datetime] = None
    total_count: Optional[int] = None
    state: Optional[int] = 0


class DataSourceUpdateDTO(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    upto: Optional[str] = None
    total_count: Optional[int] = None
    state: Optional[int] = 0


class DataSourceDTO(DataSourceCreateDTO):
    id: int
    date_created: Optional[datetime] = None
    date_modified: Optional[datetime] = None
    date_archived: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
