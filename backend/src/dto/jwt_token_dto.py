from datetime import datetime

from pydantic import BaseModel


class LoginDTO(BaseModel):
    username: str
    password: str


class JWTTokenDTO(BaseModel):
    access_token: str
    token_type: str
    expire: datetime


class JWTTokenPayloadDTO(BaseModel):
    username: str
    role: int
