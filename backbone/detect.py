import torch
import os


class Detector():
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path=os.environ.get('MODEL_PATH', 'weights/best.pt'))

    def detect(self, img):
        """
        img: PIL image
        """
        results = self.model(img)
        return results.pandas().xyxy[0]
