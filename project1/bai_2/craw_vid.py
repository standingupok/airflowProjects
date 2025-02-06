# File: dags/video_crawl_face_detect.py

from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import yt_dlp
import cv2

# Constants
TRENDING_URLS = [
    "https://www.youtube.com/results?search_query=%23shorts",
    # Add TikTok/YouTube trend URLs or implement a scraper for trends
]
SAVE_DIR = "/home/hlong/input_folder"
FRAMES_DIR = "/home/hlong/output_folder"

def crawl_videos():
    """Crawler 5 videos từ TikTok/YouTube trends."""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(SAVE_DIR, '%(title)s.%(ext)s'),
        'noplaylist': True,  # Tắt chế độ tải danh sách phát
        'playlist_items': '1-5',  # Chỉ tải 5 video đầu tiên từ danh sách phát
        'max_downloads': 5,  # Giới hạn tối đa 5 video
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in TRENDING_URLS[:5]:  # Chỉ xử lý tối đa 5 URL
            try:
                ydl.download([url])
            except yt_dlp.utils.MaxDownloadsReached:
                print("Maximum number of downloads reached. Stopping downloads.")
                break  # Dừng tải khi đạt giới hạn
            except yt_dlp.utils.DownloadError as e:
                print(f"Error downloading video from {url}: {e}")

def detect_faces():
    """Phát hiện khuôn mặt từ video đã crawler."""
    today = datetime.now().strftime("%d_%m_%y")
    day_frames_dir = os.path.join(FRAMES_DIR, today)
    if not os.path.exists(day_frames_dir):
        os.makedirs(day_frames_dir)
    
    for video_file in os.listdir(SAVE_DIR):
        video_path = os.path.join(SAVE_DIR, video_file)
        if not os.path.isfile(video_path):
            continue

        # Open video file
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps // 5)  # Process at 5 FPS

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                # Detect faces in frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) > 0:
                    # Save frame
                    frame_name = f"{os.path.splitext(video_file)[0]}_frame_{frame_count}.jpg"
                    frame_path = os.path.join(day_frames_dir, frame_name)
                    cv2.imwrite(frame_path, frame)
            frame_count += 1
        cap.release()

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'crawl_and_detect_faces',
    default_args=default_args,
    description='Crawl trending videos and detect faces daily',
    schedule_interval='0 7 * * *',  # Run at 7:00 AM every day
    start_date=datetime(2023, 12, 27),
    catchup=False,
)

t1 = PythonOperator(
    task_id='crawl_videos',
    python_callable=crawl_videos,
    dag=dag,
)

t2 = PythonOperator(
    task_id='detect_faces',
    python_callable=detect_faces,
    dag=dag,
)

t1 >> t2
