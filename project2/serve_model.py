# File: serve_model.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

# Load YOLO model (this will be replaced dynamically by the Airflow pipeline)
model = YOLO("/home/hlong/tomatopj/best.pt")

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read and convert image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    #print(image)
    
    # Predict
    results = model(image)
    predictions = results[0].boxes.xywh  # Example: Extracting bounding boxes
    #print(predictions)
    
    # Format response
    predictions_list = [
        {"x": float(box[0]), "y": float(box[1]), "w": float(box[2]), "h": float(box[3])}
        for box in predictions
    ]
    #print(predictions_list)
    return JSONResponse(content={"predictions": predictions_list})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
