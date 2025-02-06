# File: train_and_deploy_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import shutil
import os

# Path setup
BASE_DETECT_DIR = "/home/hlong/runs/detect"
API_SCRIPT_PATH = "/home/hlong/tomatopj/serve_model.py"
TEST_IMAGES_PATH = "/home/hlong/tomatopj/testimgs"

def find_latest_best_model(base_dir=BASE_DETECT_DIR):
    """Find the latest trained YOLO model."""
    # Get list of subdirectories
    subdirs = [
        os.path.join(base_dir, d) for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    # Sort subdirectories by modification time (newest first)
    subdirs = sorted(subdirs, key=os.path.getmtime, reverse=True)
    # Search for best.pt in the latest folder
    for subdir in subdirs:
        best_model_path = os.path.join(subdir, "weights", "best.pt")
        if os.path.exists(best_model_path):
            return best_model_path.replace('\\', '/')
    raise FileNotFoundError("No best.pt file found in detect directory.")

def train_model():
    """Train YOLO model."""
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.train(data="/home/hlong/tomatopj/data.yaml", epochs=3, batch=4, imgsz=320)
    print(f"Model saved to {BASE_DETECT_DIR}")

def deploy_model():
    """Deploy new model to FastAPI."""
    best_model_path = find_latest_best_model()
    shutil.copy(best_model_path, "/home/hlong/tomatopj/best.pt")
    subprocess.run(["pkill", "-f", "serve_model.py"])  # Stop old API
    subprocess.Popen(["python3", API_SCRIPT_PATH])    # Start new API
    print("API deployed with new model.")
    import time
    time.sleep(10)

def test_api():
    """Test API with prepared images."""
    import requests
    import os

    url = "http://localhost:8000/predict/"
    for image_file in os.listdir(TEST_IMAGES_PATH):
        with open(f"{TEST_IMAGES_PATH}/{image_file}", "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)
            print(f"Response for {image_file}: {response.json()}")

# Define DAG
with DAG(
    "train_and_deploy_yolo",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    train_task = PythonOperator(task_id="train_model", python_callable=train_model)
    deploy_task = PythonOperator(task_id="deploy_model", python_callable=deploy_model)
    test_task = PythonOperator(task_id="test_api", python_callable=test_api)

    train_task >> deploy_task >> test_task