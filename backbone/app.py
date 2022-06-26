
"""
Run a rest API exposing the yolov5s object detection model
"""
import io
from flask import Flask, request
from PIL import Image
from detect import Detector
import os

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5s"

detector = Detector()
@app.route(os.path.join(DETECTION_URL,'detect'), methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()

        img = Image.open(io.BytesIO(image_bytes))
        results = detector.detect(img)
        return results.to_json(orient="records")


@app.route(os.path.join(DETECTION_URL,'set-model'), methods=["POST"])
def set_model():
    if not request.method == "POST":
        return
    model_path = request.form.get("model_path")
    detector.set_model(model_path)
    return "Model set"


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=os.environ.get("PORT",5000))