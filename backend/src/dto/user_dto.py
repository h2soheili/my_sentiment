from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class UserCreateDTO(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    role: Optional[int] = 0
    state: Optional[int] = 0


class UserUpdateDTO(BaseModel):
    password: Optional[str] = None
    role: Optional[int] = 0
    state: Optional[int] = 0


class UserDTO(UserCreateDTO):
    id: int
    date_created: Optional[datetime] = None
    date_modified: Optional[datetime] = None
    date_archived: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
