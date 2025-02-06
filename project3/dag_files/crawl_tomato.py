from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from bing_image_downloader import downloader
import os

def crawl_tomato_images():
    # Lấy timestamp hiện tại để tạo folder
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = f'/home/hlong/tomatopj2/data_lake/{timestamp}'
    
    # Tạo folder mới với timestamp
    os.makedirs(output_dir, exist_ok=True)
    
    downloader.download(
        "tomato",
        limit=50,
        output_dir=output_dir,
        adult_filter_off=True,
        force_replace=False,
        timeout=60,
        verbose=True
    )

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'crawl_tomato_images_dag',
    default_args=default_args,
    description='A DAG to crawl tomato images from Bing',
    schedule_interval='0 9 * * *',  # Every day at 9 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

crawl_images_task = PythonOperator(
    task_id='crawl_tomato_images',
    python_callable=crawl_tomato_images,
    dag=dag,
)
