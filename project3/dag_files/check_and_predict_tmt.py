from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import os
import requests
import json
import time
import subprocess

DATA_LAKE_BASE_PATH = "/home/hlong/tomatopj2/data_lake"
DATA_POOL_BASE_PATH = "/home/hlong/tomatopj2/data_pool"
PREDICTION_API_URL = "http://localhost:8000/predict/"
API_SCRIPT_PATH = "/home/hlong/tomatopj2/serve_model.py"

# Tạo thư mục data_pool nếu chưa có
os.makedirs(DATA_POOL_BASE_PATH, exist_ok=True)

def deploy_api():
    """Dừng API cũ và khởi chạy API mới."""
    subprocess.run(["pkill", "-f", "serve_model.py"])  # Dừng API cũ
    subprocess.Popen(["python3", API_SCRIPT_PATH])    # Khởi chạy API mới
    print("API deployed with new model.")
    time.sleep(10)  # Chờ API khởi động

def check_and_predict():
    # Duyệt qua tất cả các thư mục trong data_lake (bao gồm sub-folder "tomato")
    lake_folders = [
        os.path.join(DATA_LAKE_BASE_PATH, folder, "tomato") for folder in os.listdir(DATA_LAKE_BASE_PATH)
        if os.path.isdir(os.path.join(DATA_LAKE_BASE_PATH, folder, "tomato"))
    ]
    # check before predict
    if(len(lake_folder) == 0):
        print("No folders found in data_lake. Skipping predicting.")
        return
    
    for lake_folder in lake_folders:
        # Lấy tên thư mục lake gốc (vd: "tomato_2024-01-01_12-30-00")
        folder_name = os.path.basename(os.path.dirname(lake_folder))
        
        # Tạo thư mục trung gian trong data_pool
        pool_folder = os.path.join(DATA_POOL_BASE_PATH, folder_name)
        os.makedirs(pool_folder, exist_ok=True)
        
        # Lấy danh sách ảnh trong lake_folder
        images = [
            os.path.join(lake_folder, f) for f in os.listdir(lake_folder)
            if os.path.isfile(os.path.join(lake_folder, f))
        ]
        
        # Dự đoán và lưu kết quả
        for image_path in images:
            with open(image_path, "rb") as f:
                files = {"file": f}
                response = requests.post(PREDICTION_API_URL, files=files)

            if response.status_code == 200:
                predictions = response.json().get("predictions", [])
                # Lưu kết quả vào thư mục trung gian
                image_name = os.path.basename(image_path)  # Ví dụ: "image_1.jpg"
                label_file = os.path.join(pool_folder, f"{image_name}.json")
                with open(label_file, "w") as label_out:
                    json.dump(predictions, label_out)
                print(f"Saved predictions for {image_path} to {label_file}")
            else:
                print(f"Failed to process {image_path}. Response: {response.status_code}, {response.text}")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'detect_tomato_images_dag',
    default_args=default_args,
    description='A DAG to detect tomatoes from images in data_lake and save labels in organized data_pool folders',
    schedule_interval='@hourly',  # Chạy mỗi giờ
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

deploy_api_task = PythonOperator(
    task_id='deploy_api',
    python_callable=deploy_api,
    dag=dag,
)

check_and_predict_task = PythonOperator(
    task_id='check_and_predict',
    python_callable=check_and_predict,
    dag=dag,
)

deploy_api_task >> check_and_predict_task
