import torch
from train.yolov7.models.common import autoShape
from train.yolov7.utils.torch_utils import select_device

class Detector():
    def set_model(self, model_path, device=None):
        self.device = select_device(device)
        self.model = autoShape(model)
        self.model.to(self.device)

    def detect(self, img):
        """
        img: PIL image
        """
        results = self.model(img)
        return results.pandas().xyxy[0]

    def detect_render(self, img):
        """
        img: PIL image
        """
        results = self.model(img)
        return results.render()[0]
