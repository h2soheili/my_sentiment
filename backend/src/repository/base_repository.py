from datetime import datetime
from typing import Any, Generic, List, Optional, Type, TypeVar

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from retry_async import retry
from sqlalchemy import text, and_, Sequence
from sqlalchemy.future import select
from sqlalchemy.orm import class_mapper

from backend.src.constants import States
from backend.src.core import async_db_session
from backend.src.core import log_factory
from backend.src.models import BaseEntity

ModelEntity = TypeVar("ModelEntity", bound=BaseEntity)
CreateDTO = TypeVar("CreateDTO", bound=BaseModel)
UpdateDTO = TypeVar("UpdateDTO", bound=BaseModel)

logger = log_factory.get_logger(__name__)


class BaseRepository(Generic[ModelEntity, CreateDTO, UpdateDTO]):
    def __init__(self, model: Type[ModelEntity]):
        """
        CRUD object with default methods to Create, Read, Update, Delete (CRUD).

        **Parameters**

        * `model`: A SQLAlchemy model class
        * `schema`: A Pydantic model (schema) class
        """
        self.model = model

    def with_nolock(self, query):
        # query = query.with_hint(self.model, 'WITH (NOLOCK)')
        # query = query.with_lockmode("read_nowait")
        return query


    async def get(self, id: Any, only_actives: bool = True) -> Optional[ModelEntity]:
        async with async_db_session() as session:
            query = select(self.model)
            query = query.where(self.model.id == id)
            if only_actives:
                query = query.where(self.model.state == States.Active.value)
            query = self.with_nolock(query)
            query = await session.execute(query)
            return query.scalars().first()


    async def get_with_key(self, key: Any, only_actives: bool = True) -> Optional[ModelEntity]:
        async with async_db_session() as session:
            query = select(self.model)
            query = query.where(self.model.key == key)
            if only_actives:
                query = query.where(self.model.state == States.Active.value)
            query = self.with_nolock(query)
            query = await session.execute(query)
            return query.scalars().first()

    def filter_only_actives(self, query, only_actives: bool = True):
        if only_actives:
            query = query.where(self.model.state == States.Active.value)
        return query

    def filter_by_user_id(self, query, user_id: int = None):
        if user_id is not None and vars(self.model).get("time"):
            query = query.filter(self.model.user_id == user_id)
        return query

    def filter_by_time(self, query,
                       after: datetime = None,
                       before: datetime = None, ):
        if (after or before) and vars(self.model).get("time"):
            if after and before:
                query = query.filter(and_(self.model.time >= after, self.model.time <= before))
            else:
                if after:
                    query = query.where(self.model.time >= after)
                elif before:
                    query = query.where(self.model.time <= before)
        return query


    async def get_multi(self,
                        key: Any = None,
                        id: Any = None,
                        after: datetime = None,
                        before: datetime = None,
                        time: datetime = None,
                        skip: int = 0,
                        limit: int = 100,
                        desc: bool = True,
                        only_actives: bool = True,
                        user_id: int = None
                        ) -> Sequence[ModelEntity]:

        async with async_db_session() as session:
            query = select(self.model)
            query = self.filter_only_actives(query, only_actives)
            query = self.filter_by_time(query, after, before)
            query = self.filter_by_user_id(query, user_id)
            order_by = "desc" if desc else "asc"
            query = query.order_by(text(f"id {order_by}"))
            query = query.offset(skip).limit(limit)
            query = self.with_nolock(query)
            query = await session.execute(query)
            query = query.scalars().all()
            return query


    async def create(self, db_obj: CreateDTO, ) -> Optional[ModelEntity]:
        async with async_db_session() as session:
            try:
                # print("create")
                # print(db_obj)
                # logger.info(db_obj)
                db_obj = jsonable_encoder(db_obj)
                # obj_in_data = obj_in
                # print(1)
                db_obj = self.model(**db_obj)
                session.add(db_obj)
                await session.commit()
                await session.refresh(db_obj)
                return db_obj
            except Exception as error:
                await session.rollback()
                logger.error("model create error", error=error, db_obj=db_obj)
                raise error
                # raise HTTPException(
                #     status_code=500,
                #     detail=str(error)
                # )


    async def update(
            self,
            db_obj: ModelEntity,
            obj_in: UpdateDTO
    ) -> ModelEntity:
        async with async_db_session() as session:

            try:
                # print("update")
                # print(db_obj)
                # print(obj_in)
                db_obj = jsonable_encoder(db_obj)
                # obj_in = obj_in.dict()
                # obj_data = db_obj.__dict__
                # obj_data = obj_in
                update_data = {}
                if isinstance(obj_in, dict):
                    update_data = obj_in
                else:
                    try:
                        update_data = obj_in.model_dump(exclude_unset=True)
                    except Exception as e:
                        try:
                            update_data = obj_in.__dict__
                            # if "_sa_instance_state" in update_data:
                            #     update_data.pop("_sa_instance_state")
                        except Exception as ee:
                            pass

                # update_data = jsonable_encoder(update_data)

                # if "id" in update_data:
                #     update_data.pop("id")
                # for field in obj_data:
                #     if field in update_data:
                #         setattr(db_obj, field, update_data[field])
                # print('update_data')
                # print(update_data)
                for field in update_data:
                    if field not in ("_sa_instance_state", "id"):
                        setattr(db_obj, field, update_data[field])
                # db_obj = self.model(**db_obj)
                # print('db_obj', db_obj)
                # print(db_obj.serialize())
                session.add(db_obj)
                await session.commit()
                await session.refresh(db_obj)
                return db_obj
            except Exception as error:
                await session.rollback()
                logger.error("model update error", error=error, db_obj=db_obj.serialize(), obj_in=obj_in)
                raise error
                # raise HTTPException(
                #     status_code=500,
                #     detail=str(error)
                # )


    async def remove(self, id: Any) -> Optional[ModelEntity]:
        async with async_db_session() as session:
            try:
                obj = await session.execute(select(self.model).get(id))
                await session.delete(obj)
                await session.commit()
                await session.refresh(obj)
                return obj
            except Exception as error:
                await session.rollback()
                logger.error("model remove error", error=error)
                raise error
                # raise HTTPException(
                #     status_code=500,
                #     detail=str(error)
                # )


    @staticmethod
    def object_to_dict(obj: ModelEntity, found=None) -> dict:
        if found is None:
            found = set()
        mapper = class_mapper(obj.__class__)
        columns = [column.key for column in mapper.columns]
        get_key_value = lambda c: (c, getattr(obj, c).isoformat()) if isinstance(getattr(obj, c), datetime) else (
            c, getattr(obj, c))
        out = dict(map(get_key_value, columns))
        for name, relation in mapper.relationships.items():
            if relation not in found:
                found.add(relation)
                related_obj = getattr(obj, name)
                if related_obj is not None:
                    if relation.uselist:
                        out[name] = [BaseRepository.object_to_dict(child, found) for child in related_obj]
                    else:
                        out[name] = BaseRepository.object_to_dict(related_obj, found)
        return out

    @staticmethod
    def list_of_object_to_list_of_dict(rows: List[ModelEntity], ) -> List[dict]:
        return [BaseRepository.object_to_dict(row) for row in rows]
