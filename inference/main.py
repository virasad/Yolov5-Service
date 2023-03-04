
"""
Run a rest API exposing the yolov5s object detection model
"""
import io
from fastapi import FastAPI, File, UploadFile, Request, Form
from PIL import Image
from detect import Detector
import os
import numpy as np
import cv2
from starlette.responses import StreamingResponse
import json

app = FastAPI(title="YOLOv5 Service")

DETECTION_URL = "/v1/object-detection/yolov5s"

detector = Detector()

def generate_random_colors(n):
    if n == 1:
        return [(255, 0, 0)]
    colors = []
    colors.append((0, 0, 0))
    for i in range(1, n):
        colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
    return colors


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


color_clases = {
    1:(0,150,0),
    2:(150,0,0),
    3:(0,0,150),
    0:(150,150,0),
    4:(250,100,0),
    5:(0,250,100),
    6:(120,120,120),
    7:(250,0,100),
    8:(120, 50, 250),
    9:(50, 250, 100),
    10:(75, 175, 95)
    }

@app.post("/predict-image")
async def predict(image: UploadFile = File(...)):
    # try:
        contents = await image.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        response = detector.detect(img)
        print(type(response.to_json(orient="records")))
        res = json.loads(response.to_json(orient="records"))
        bbox_list = []
        for i in res:
            print(i)
            x0 = i['xmin']
            y0 = i['ymin']
            x1 = i['xmax'] 
            y1 = i['ymax']
            bbox_list.append([x0, y0, x1, y1, i['name'], i['confidence'], i['class']])
            bbox = [x0, y0, x1 - x0, y1 - y0]
                
        for bbox in bbox_list:
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_clases[bbox[6]], 4)
            text = f'{bbox[4]} {"{:.2f}".format(bbox[5])}'
            font                   = cv2.FONT_HERSHEY_COMPLEX
            fontScale              = 1
            fontColor              = (255,255,255)
            thickness              = 1
            lineType               = 1
            text_size, _ = cv2.getTextSize(text, font, fontScale, thickness)
            text_w, text_h = text_size
            bottomLeftCornerOfText = (int((bbox[0])),int(bbox[1]))
            x, y = bottomLeftCornerOfText
            cv2.rectangle(img, bottomLeftCornerOfText, (x + text_w, y - text_h), color_clases[bbox[6]], -1)
            cv2.putText(img, 
                text, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
        img_encoded = cv2.imencode('.jpg', img)[1]
        img_bytes = img_encoded.tobytes()

        return StreamingResponse(io.BytesIO(img_bytes), media_type='image/jpeg')


    # except Exception as e:
    #     return {"message": str(e)}