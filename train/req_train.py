import json
import requests
from pathlib import Path


def train_yolo_data(is_augment):
    url = 'http://127.0.0.1:8000/train'  # server url
    data = {'label': ['c1', 'c2'],  # one class
            'image_path': '/dataset/images',
            'label_path': '/dataset/labels',
            'image_size': 640,
            'epochs': 7,
            'weight': 'yolov5n.pt',
            'validation_split': 0.2,
            'data_type': 'yolo',
            'save_dir': 'results/',
            'is_augment': is_augment}
    r = requests.post(url, json=data)
    return r.text


def train_coco_data(is_augment):
    url = 'http://127.0.0.1:8000/train'  # server url

    path = str((Path(__file__).parent).parent) + \
        '/volumes/dataset/labels/coco_annotation.json'
    with open(path) as f:
        coco_annotation = json.load(f)
    data = {'label': coco_annotation,
            'image_path': '/dataset/images',
            'image_size': 640,
            'epochs': 7,
            'weight': 'yolov5n.pt',
            'validation_split': 0.1,
            'data_type': 'coco',
            'save_dir': 'results/',
            'is_augment': is_augment}
    r = requests.post(url, json=data)
    return r.text


if __name__ == '__main__':
    data_type = 'coco'  # 'yolo' or 'coco'
    is_augment = True  # True or False
    if data_type == 'yolo':
        train_yolo_data(is_augment)
    elif data_type == 'coco':
        train_coco_data(is_augment)
