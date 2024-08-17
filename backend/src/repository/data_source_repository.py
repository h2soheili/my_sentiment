from backend.src.repository.base_repository import BaseRepository
from backend.src.models import DataSourceEntity
from backend.src.dto import DataSourceCreateDTO, DataSourceUpdateDTO


class DataSourceRepository(BaseRepository[DataSourceEntity, DataSourceCreateDTO, DataSourceUpdateDTO]):
    pass


data_source_repository = DataSourceRepository(DataSourceEntity)
