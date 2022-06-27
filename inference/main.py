
"""
Run a rest API exposing the yolov5s object detection model
"""
import io
from fastapi import FastAPI, File, UploadFile, Request, Form
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


@app.post(os.path.join(DETECTION_URL, 'set-model'))
async def set_model(request: Request, model_path: str = Form(...)):
    form_data = await request.form()
    model_path = form_data.get('model_path')
    detector.set_model(model_path)
    return {'message': 'ok'}
