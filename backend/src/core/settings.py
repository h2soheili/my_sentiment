from typing import List
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.src.core.global_log import log_factory

logger = log_factory.get_logger(__name__)


pwd = os.getcwd()
# is_colab = pwd == '/content'
# p = pwd.replace('ai', '').replace('csv_files', '').strip() if not is_colab else "/content/algo_runner"
# env_file = os.path.join(p, '.env')
env_file = "../.env"
env_file = os.path.abspath(env_file)
print('Config env_file_path is:', env_file)
print("os.getcwd() - >>>> ", os.getcwd())


class Settings(BaseSettings):
    SECRET_KEY: str = "secret"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 50
    BACKEND_CORS_ORIGINS: List[str] = []
    SQL_DSN: str = ""
    MONGO_URL: str = ""
    model_config = SettingsConfigDict(env_file=env_file, env_file_encoding='utf-8', extra='allow')


settings = Settings()
print(settings)
