import json
import requests


def train_yolo_data(is_augment):
    with open("train_data.json") as f:
        train_data = json.load(f)
    url = train_data["url"]  # server url
    data = train_data["data"]
    r = requests.post(url, json=data)
    return r.text


def train_coco_data(is_augment):
    with open("train_data.json") as f:
        train_data = json.load(f)
    url = train_data["url"]  # server url
    data = train_data["data"]
    r = requests.post(url, params=data)
    return r.text


if __name__ == "__main__":  # 'yolo' or 'coco'
    is_augment = True  # True or False
    train_coco_data(is_augment)
