pytorch + tiktoken + fastapi + kafka + postgres 

------

for install sqlalchemy 2 async  
```pip install sqlalchemy[asyncio]``` or ```pip install sqlalchemy"[asyncio]" ```

------



##Migration And Database:

#initial data   
run `initial_data.py` script

generate migration(root folder is logic/app): `alembic revision --autogenerate -m "new tables"` <br>
apply migrations(root folder is logic/app): `alembic upgrade head`

* If you created a new model in `./models/`, make sure to import it in `./alembic/env.py` and `./db/init_db.py`,
  these Python modules that imports all the models will be used by Alembic.

* After changing a model (for example, adding a column), inside the container, create a revision, e.g.:

```console
$ alembic revision --autogenerate -m "Add column last_name to User model"
```

* Commit to the git repository the files generated in the alembic directory.

* After creating the revision, run the migration in the database (this is what will actually change the database):

```console
$ alembic upgrade head
```

import new models in `models/__init__.py` to be know for alembic

##
