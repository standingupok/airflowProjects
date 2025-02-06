from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from ultralytics import YOLO
import subprocess
import shutil
import os

DATA_POOL_PATH = "/home/hlong/tomatopj2/data_pool"
API_MODEL_PATH = "/home/hlong/tomatopj2/best.pt"
API_SCRIPT_PATH = "/home/hlong/tomatopj2/serve_model.py"
BASE_DETECT_DIR = "/home/hlong/runs/detect"  # Thư mục chứa kết quả huấn luyện YOLO

def check_and_train():
    """Kiểm tra data_pool và huấn luyện model nếu có đủ dữ liệu.
    images = [
        os.path.join(DATA_POOL_PATH, f) for f in os.listdir(DATA_POOL_PATH)
        if f.endswith(".json")  # Chỉ kiểm tra file JSON chứa nhãn
    ]
    if len(images) < 50:
        print("Not enough data for training. Skipping.")
        return"""
        
    """Kiểm tra data_pool và huấn luyện model nếu có ít nhất một thư mục - 1 Folder có 50 labels"""
    pool_folders = [
        os.path.join(DATA_POOL_PATH, folder) for folder in os.listdir(DATA_POOL_PATH)
        if os.path.isdir(os.path.join(DATA_POOL_PATH, folder))
    ]

    if len(pool_folders) == 0:  # Không có thư mục nào trong data_pool
        print("No folders found in data_pool. Skipping training.")
        return
        
    # Huấn luyện mô hình mới
    print("Starting model training...")
    model = YOLO("yolo11n.pt")
    model.train(data="/home/hlong/tomatopj2/data.yaml", epochs=3, batch=4, imgsz=320)
    print(f"Model training completed. Results saved to {BASE_DETECT_DIR}")

def find_latest_best_model():
    """Tìm model tốt nhất mới được huấn luyện."""
    subdirs = [
        os.path.join(BASE_DETECT_DIR, d) for d in os.listdir(BASE_DETECT_DIR)
        if os.path.isdir(os.path.join(BASE_DETECT_DIR, d))
    ]
    subdirs = sorted(subdirs, key=os.path.getmtime, reverse=True)
    for subdir in subdirs:
        best_model_path = os.path.join(subdir, "weights", "best.pt")
        if os.path.exists(best_model_path):
            return best_model_path.replace('\\', '/')
    raise FileNotFoundError("No best.pt file found in detect directory.")

def compare_and_replace():
    """So sánh model mới với model API và thay thế nếu tốt hơn."""
    print("Comparing new model with current API model...")
    # Tìm model mới nhất
    new_model_path = find_latest_best_model()
    print(f"New model path: {new_model_path}")
    
    # Đánh giá model mới
    new_model = YOLO(new_model_path)
    new_results = new_model.val(data="/home/hlong/check_acc_val/data.yaml")
    new_map = new_results.box.map
    print(f"New model mAP: {new_map}")

    # Đánh giá model hiện tại trên API
    current_model = YOLO(API_MODEL_PATH)
    current_results = current_model.val(data="/home/hlong/check_acc_val/data.yaml")
    current_map = current_results.box.map
    print(f"Current API model mAP: {current_map}")

    # So sánh và thay thế nếu cần
    if new_map > current_map:
        print("New model is better. Replacing API model.")
        shutil.copy(new_model_path, "/home/hlong/tomatopj2/best.pt")
        subprocess.run(["pkill", "-f", "serve_model.py"])  # Stop old API
        subprocess.Popen(["python3", API_SCRIPT_PATH])    # Start new API
    else:
        print("Current API model is better. No replacement made.")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'train_and_update_model_dag',
    default_args=default_args,
    description='A DAG to train a new model if data_pool has enough data and update API model if the new model is better',
    schedule_interval='@hourly',  # Chạy mỗi giờ
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

check_and_train_task = PythonOperator(
    task_id='check_and_train',
    python_callable=check_and_train,
    dag=dag,
)

compare_and_replace_task = PythonOperator(
    task_id='compare_and_replace',
    python_callable=compare_and_replace,
    dag=dag,
)

check_and_train_task >> compare_and_replace_task