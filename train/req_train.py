import json
import requests

# fisrt Copy train2.py in yolov5 folder
# then run this script


def train_yolo_data_without_augment():
    url = 'http://127.0.0.1:8000/train'  # server url
    data = {'label': ['c1', 'c2'],
            'image_path': '/code/dataset/images',
            'label_path': '/code/dataset/labels',
            'image_size': 640,
            'epochs': 7,
            'weight': 'yolov5n.pt',
            'validation_split': 0.2,
            'data_type': 'yolo',
            'save_dir': 'results/'}
    r = requests.post(url, json=data)
    print(r.text)


# function train yolo data with augment
def train_yolo_data_with_augment():
    url = 'http://127.0.0.1:8000/train'  # server url
    data = {'label': ['c1', 'c2'],
            'image_path': 'data/images',
            'label_path': 'data/labels',
            'image_size': 640,
            'epochs': 7,
            'weight': 'yolov5n.pt',
            'validation_split': 0.2,
            'data_type': 'yolo',
            'save_dir': 'results/',
            'is_augment': True,
            'augment_params': {'count_of_each': 2}}
    r = requests.post(url, json=data)
    print(r.text)


def train_coco_data_without_augment():
    url = 'http://127.0.0.1:8000/train'  # server url

    with open('coco_annotation.json') as f:
        coco_annotation = json.load(f)
        coco_annotation = json.dumps(coco_annotation)

    data = {'label': coco_annotation,
            'image_path': 'image_path',
            'label_path': 'label_path',
            'image_size': 640,
            'epochs': 7,
            'weight': 'yolov5n.pt',
            'validation_split': 0.1,
            'data_type': 'coco',
            'save_dir': 'save_dir'}
    r = requests.post(url, json=data)
    print(r.text)


def train_coco_data_with_augment():
    url = 'http://127.0.0.1:8000/train'  # server url

    with open('coco_annotation.json') as f:
        coco_annotation = json.load(f)
        coco_annotation = json.dumps(coco_annotation)

    data = {'label': coco_annotation,
            'image_path': 'image_path',
            'label_path': 'label_path',
            'image_size': 640,
            'epochs': 7,
            'weight': 'yolov5n.pt',
            'validation_split': 0.1,
            'data_type': 'coco',
            'save_dir': 'save_dir',
            'is_augment': True,
            'augment_params': {'count_of_each': 2}}
    r = requests.post(url, json=data)
    print(r.text)


if __name__ == '__main__':
    data_type = 'yolo'  # 'yolo' or 'coco'
    is_augment = False  # True or False
    if data_type == 'yolo':
        if is_augment:
            train_yolo_data_with_augment()  # train yolo data with augment
        else:
            train_yolo_data_without_augment()  # train yolo data without augment
    elif data_type == 'coco':
        if is_augment:
            train_coco_data_with_augment()  # train coco data with augment
        else:
            train_coco_data_without_augment()  # train coco data without augment
