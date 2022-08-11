import json
import os
import shutil

import requests
from fastapi import FastAPI

from util.coco2yolo import COCO2YOLO
from util.train_validation_sep import separate_test_val
from util.util import jsonfile2dict
from util.yaml_file import dict_to_yaml
from yolov5 import train

normpath = os.path.normpath

app = FastAPI(
    title="Yolov5 Train API",
    description="Api for segmentation model training",
    version="0.1.0",
    contact={
        "name": "Virasad",
        "url": "https://virasad.ir",
        "email": "info@virasad.ir",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


@app.get("/")
def root():
    return {"message": "Welcome to the Train API get documentation at /docs"}


@app.post('/train')
async def train_model(labels: str,
                      images_path: str,
                      image_size: int = 640,
                      epochs: int = 100,
                      weight: str = "yolov5n.pt",
                      validation_split: float = 0.2,
                      data_type: str = "yolo",
                      save_dir: str = "results/",
                      batch_size: int = 2,
                      response_url: str = None,
                      log_url: str = None,
                      classes: list = None):
    if data_type not in ['yolo', 'coco']:
        return {'message': 'data_type must be yolo or coco'}
    # check images dir
    if not os.path.isdir(images_path):
        return {'message': 'image_path is not a directory'}
    # check weight
    weights_v5 = ['yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']
    weights_v6 = ['yolov5n6.pt', 'yolov5s6.pt', 'yolov5m6.pt', 'yolov5l6.pt', 'yolov5x6.pt']

    if weight not in weights_v5 + weights_v6:
        return {'message': 'weight must be one of yolov5n.pt, yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt, '
                           'yolov5n6.pt, yolov5s6.pt, yolov5m6.pt, yolov5l6.pt, yolov5x6.pt'}
        # v5 should have the image size of 640 x 640 and v6 should have the image size of 1280 x 1280
    if weight in weights_v5:
        if image_size != 640:
            return {
                'message': 'image_size must be 640 for yolov5n.pt, yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt'}

    elif weight in weights_v6:
        if image_size != 1280:
            return {
                'message': 'image_size must be 1280 for yolov5n6.pt, yolov5s6.pt, yolov5m6.pt, yolov5l6.pt, yolov5x6.pt'}
    images_path = normpath(images_path)
    labels = normpath(labels)
    dataset_dir = os.path.dirname(images_path)
    if data_type == 'coco':
        # check labels json
        if not os.path.isfile(labels):
            # check extension of label file
            if not labels.endswith('.json'):
                return {'message': 'label is not a json file'}
            return {'message': 'label_path is not a file'}
        # check if label file is valid
        try:
            with open(labels) as f:
                json.load(f)
        except json.JSONDecodeError:
            return {'message': 'label is not a valid json file'}
        labels_dir = os.path.join(dataset_dir, 'labels')
        os.makedirs(labels_dir, exist_ok=True)
        c2y = COCO2YOLO(jsonfile2dict(labels),
                        output=labels_dir)
        c2y.coco2yolo()
        classes = c2y.get_classes()
        print(classes)

    elif data_type == 'yolo':
        # check labels dir
        if not os.path.isdir(labels):
            return {'message': 'label is not a directory'}

        # check if label dir is valid
        if classes is None:
            return {'message': 'classes is required in yolo data type'}
        print(images_path, labels)
        labels_dir = labels


    train_dir = os.path.join(dataset_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    test_dir = os.path.join(dataset_dir, 'val')
    print(test_dir)
    print(train_dir)
    train_images_dir, train_txts_dir, validation_images_dir, validation_txts_dir = separate_test_val(
        images_dir=images_path,
        txts_dir=labels_dir,
        dst_validatoion_dir=train_dir,
        dst_train_dir=test_dir,
        validation_percentage=validation_split
    )

    d = {
        'train': os.path.abspath(train_images_dir),
        'val': os.path.abspath(validation_images_dir),
        'nc': len(classes),
        'names': classes
    }
    data_yml = os.path.join(dataset_dir, 'data.yml')
    dict_to_yaml(d, data_yml)
    # save_dir = normpath(req['save_dir'])
    # Training yolo
    train.run(data=data_yml, imgsz=image_size, weights=weight,
              save_dir=save_dir, epochs=epochs, batch_size=batch_size,
              project=save_dir, name='', exists_ok=True, log_url=log_url)
    # # delete temp file
    # if os.path.exists('tmp') and os.path.isdir('tmp'):
    #     shutil.rmtree('tmp')
    #
    resp_data = {
        'save_dir': save_dir,
        'model_dir': os.path.join(save_dir, 'weights/best.pt'),
        'classes': classes
    }
    # convert post_data to json
    resp_data = json.dumps(resp_data)
    # get end_url from os environment variable

    if response_url:
        requests.post(response_url, json=resp_data)
    return resp_data
