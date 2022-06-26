
"""
Run a rest API exposing the yolov5s object detection model
"""
import io
from fastapi import FastAPI, File, UploadFile
import uvicorn
from PIL import Image
from detect import Detector
import os

app = FastAPI(title="YOLOv5 Service")

DETECTION_URL = "/v1/object-detection/yolov5s"

detector = Detector()


@app.post(os.path.join(DETECTION_URL, 'detect'))
async def predict(image: UploadFile = File(...)):

    contents = await image.read()
    img = Image.open(io.BytesIO(contents))
    results = detector.detect(img)
    return results.to_json(orient="records")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
