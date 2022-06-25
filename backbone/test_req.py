import requests
import pprint
import cv2
import sys


DETECTION_URL = "http://localhost:5000/v1/object-detection/yolov5s/detect"


def send_with_image(image_path, detection_url):
    image_data = open(image_path, "rb").read()
    response = requests.post(detection_url, files={"image": image_data}).json()
    pprint.pprint(response)
    return response


def send_cv2_image(cv2image, detection_url):
    cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
    img_bytes = cv2.imencode('.jpg', cv2image)[1].tobytes()
    response = requests.post(detection_url, files={"image": img_bytes}).json()
    pprint.pprint(response)


if __name__ == "__main__":
    cv_image = cv2.imread(sys.argv[1])
    send_cv2_image(cv_image, DETECTION_URL)
