import unicodedata

from sqlalchemy import create_engine, text, URL
from sqlalchemy.orm import scoped_session, sessionmaker

if __name__ == "__main__":
    print(1)
    # url_object = URL.create(
    #     "postgresql",
    #     username="admin",
    #     password="pass1234@",
    #     host="localhost",
    #     port=5432,
    #     database="nlp",
    # )
    # engine = create_engine(url=url_object, pool_size=10, max_overflow=20)
    #
    # session = scoped_session(sessionmaker(bind=engine))()
    #
    # print(session.execute(text("select count(*) from temp_table")))
    city = "â توان نیروی"

    normalized = unicodedata.normalize('NFD', city)
    new_city = u"".join([c for c in normalized if not unicodedata.combining(c)])
    print(new_city)