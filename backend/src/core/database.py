from sqlalchemy import URL
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from backend.src.core import log_factory

# from sqlalchemy import event
# from sqlalchemy.engine import Engine
# import time
#
#
# @event.listens_for(Engine, "before_cursor_execute")
# def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
#     conn.info.setdefault("query_start_time", []).append(time.time())
#     # logger.debug("Start Query: ", statement=statement)
#
#
# @event.listens_for(Engine, "after_cursor_execute")
# def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
#     total = time.time() - conn.info["query_start_time"].pop(-1)
#     logger.debug(f"Query Complete! Total Time: {total}", statement=statement)


logger = log_factory.get_logger(__name__)

db_engine = None


def async_db_session() -> AsyncSession:
    try:

        global db_engine
        if db_engine is None:
            url_object = URL.create(
                "postgresql+asyncpg",
                username="admin",
                password="pass1234@",
                host="localhost",
                port=5432,
                database="nlp",
            )
            db_engine = create_async_engine(url=url_object, pool_pre_ping=True, poolclass=StaticPool)
        async_session = async_sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=db_engine)()
        return async_session
    except Exception as e:
        logger.error("async_db_session", error=e)
