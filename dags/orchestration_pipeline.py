from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
from processing import (
    fetch_and_upload_sentiweb_data,
    fetch_and_upload_meteo_data,
    transform_and_upload_data,
    train_lightgbm_and_log,
    predict_and_upload
)

default_args = {
    'owner':'airflow',
    'start_date': datetime(2025,1,1),
    'retries':1,
    'retry_delay': timedelta(minutes=5)
}

with DAG("global_pipeline_complete_s3_auto", default_args=default_args,
         schedule_interval='0 0 * * 4', catchup=False) as dag:

    # Sentiweb
    with TaskGroup("sentiweb_branch") as sentiweb_group:
        fetch_senti = PythonOperator(
            task_id="fetch_and_upload_sentiweb_data",
            python_callable=fetch_and_upload_sentiweb_data
        )

    # Meteo
    with TaskGroup("meteo_branch") as meteo_group:
        fetch_meteo = PythonOperator(
            task_id="fetch_and_upload_meteo_data",
            python_callable=fetch_and_upload_meteo_data
        )

    # Transformation
    with TaskGroup("transformation_branch") as transform_group:
        transform_data = PythonOperator(
            task_id="transform_and_upload_data",
            python_callable=transform_and_upload_data
        )

    # Training
    with TaskGroup("training_branch") as training_group:
        train_model = PythonOperator(
            task_id="train_lightgbm",
            python_callable=train_lightgbm_and_log
        )

    # Prediction
    with TaskGroup("predict_branch") as predict_group:
        predict = PythonOperator(
            task_id="predict_and_upload",
            python_callable=predict_and_upload
        )

    # Dépendances
    [sentiweb_group, meteo_group] >> transform_group >> training_group >> predict_group

