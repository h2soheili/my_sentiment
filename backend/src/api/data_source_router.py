from typing import List

from fastapi import APIRouter, HTTPException, Depends
from starlette import status

from backend.src.core import log_factory
from backend.src.core.deps import get_current_active_user
from backend.src.dto import (DataSourceCreateDTO,
                             DataSourceUpdateDTO, DataSourceDTO)
from backend.src.models import UserEntity
from backend.src.repository import data_source_repository

router = APIRouter()

logger = log_factory.get_logger(__name__)


@router.get("/", response_model=List[DataSourceDTO])
async def read_instances(
        *,
        skip: int = 0,
        limit: int = 100,
        current_user: UserEntity = Depends(get_current_active_user),
):
    """
    Retrieve instances.
    """
    res = await data_source_repository.get_multi(skip=skip, limit=limit)
    return [DataSourceDTO(**i.serialize()) for i in res]


@router.post("/", response_model=DataSourceDTO)
async def create_instance(
        *,
        instance_in: DataSourceCreateDTO,
        current_user: UserEntity = Depends(get_current_active_user),
):
    """
    Create new instance.
    """
    res = await data_source_repository.create(instance_in)
    return DataSourceDTO(**res.serialize())


@router.put("/{_id}", response_model=DataSourceDTO)
async def update_instance(
        *,
        _id: int,
        instance_in: DataSourceUpdateDTO,
        current_user: UserEntity = Depends(get_current_active_user),
):
    """
    Update an instance.
    """
    instance = await data_source_repository.get(_id)
    if not instance:
        raise HTTPException(status_code=404, detail="instance not found")
    res = await data_source_repository.update(db_obj=instance, obj_in=instance_in)
    return DataSourceDTO(**res.serialize())


@router.get("/{_id}", response_model=DataSourceDTO)
async def read_instance(
        *,
        _id: int,
        current_user: UserEntity = Depends(get_current_active_user),
):
    """
    Get instance by ID.
    """
    res = await data_source_repository.get(_id)
    if not res:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="instance not found")
    return DataSourceDTO(**res.serialize())


@router.delete("/{_id}", response_model=DataSourceDTO)
async def delete_instance(
        *,
        _id: int,
        current_user: UserEntity = Depends(get_current_active_user),
):
    """
    Delete an instance.
    """
    res = await data_source_repository.get(_id)
    if not res:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="instance not found")
    res = await data_source_repository.remove(_id)
    return DataSourceDTO(**res.serialize())
