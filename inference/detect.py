import torch


class Detector():
    def set_model(self, model_path):
        self.model = torch.hub.load(
            "/code/yolov5/", 'custom', source='local', path=model_path)

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
