from airflow import DAG
from airflow.decorators import task


@task.virtualenv(
    task_id="news-feeder",
    # requirements=["cloudscraper~=1.2.7", "redis~=4.5.5", "bs4~=0.0.1", "apache-airflow~=2.7"],
    requirements=["cloudscraper~=1.2.7"],
    system_site_packages=True)
def do_job():
    pass


import datetime

with DAG(
        "bv-feeder-v1",
        default_args={
            "depends_on_past": False,
            "email": ["airflow@example.com"],
            "email_on_failure": False,
            "email_on_retry": False,
            "retries": 1,
            "retry_delay": datetime.timedelta(minutes=1),
            # 'queue': 'bash_queue',
            # 'pool': 'backfill',
            # 'priority_weight': 10,
            # 'end_date': datetime(2016, 1, 1),
            # 'wait_for_downstream': False,
            # 'sla': timedelta(hours=2),
            # 'execution_timeout': timedelta(seconds=300),
            # 'on_failure_callback': some_function, # or list of functions
            # 'on_success_callback': some_other_function, # or list of functions
            # 'on_retry_callback': another_function, # or list of functions
            # 'sla_miss_callback': yet_another_function, # or list of functions
            # 'trigger_rule': 'all_success'
        },
        description="A simple DAG",
        schedule="*/2 * * * *",
        catchup=False,
        tags=["bv-news-feeder"],
        doc_md="NewsFeeder MD",

) as dag:
    do_job()

# callable_virtualenv()
