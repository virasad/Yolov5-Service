import os
import shutil
from fastapi import FastAPI, Request
import uvicorn
from util.util import dict_to_json, mkdir_p, movefiles, jsonfile2dict
import json
from util.yaml_file import dict_to_yaml
from preimutils.object_detection.yolo.coco2yolo import COCO2YOLO
from preimutils.object_detection.yolo.train_validation_sep import separate_test_val
from yolov5 import train
import requests

normpath = os.path.normpath

app = FastAPI(title="Train")
TRAIN_URL = "/"


@app.post(os.path.join(TRAIN_URL, 'train'))
async def train_model(request: Request):
    req = await request.json()
    data = req["label"]
    image_path = req["image_path"]
    weight = req["weight"]
    epochs = req["epochs"]
    validation_split = req["validation_split"]
    data_type = req["data_type"]
    augment = req.get('is_augment', False)

    try:
        image_size = req["image_size"]
    except:
        image_size = 640

    try:
        Log_url = req["log_url"]
    except:
        Log_url = None

    # Create temp folder
    if not os.path.exists('tmp'):
        mkdir_p('tmp')
    else:
        shutil.rmtree('tmp')           # Removes all the subdirectories!
        mkdir_p('tmp')

    # make raw Label Folder
    os.makedirs(os.path.join('tmp/DATASET/raw_annotations/'))
    image_path = normpath(image_path)
    print(data)
    if data_type == "coco":
        # Save coco file in temp folder
        dict_to_json(data, os.path.join('tmp', 'coco.json'))
        # Load coco.json file
        c2y = COCO2YOLO(jsonfile2dict(normpath('tmp/coco.json')),
                        normpath('tmp/DATASET/raw_annotations'))
        # Save Classes in coco_classes.txt
        c2y.save_classes()

        # Copy images to tmp
        os.makedirs('tmp/DATASET/raw_images')
        images_info = c2y._load_images_info()
        for images in images_info:
            name = images_info[images][0]
            shutil.copyfile(os.path.join(image_path, name),
                            os.path.join('tmp/DATASET/raw_images', name))

        # Coco to YOLO
        c2y.coco2yolo()

        # Create Data Yaml file
        classes = c2y._categories()
        classes = list(classes.values())

    elif data_type == 'yolo':
        label_path = req["label_path"]
        classes = data
        os.makedirs('tmp/DATASET/raw_images')
        print(image_path, label_path)
        for name in os.listdir(image_path):
            shutil.copyfile(os.path.join(image_path, name), os.path.join(
                normpath('tmp/DATASET/raw_images'), name))

        for name in os.listdir(normpath(label_path)):
            shutil.copyfile(os.path.join(label_path, name), os.path.join(
                normpath('tmp/DATASET/raw_annotations'), name))

    # Train test split
    separate_test_val(
        images_dir=normpath('tmp/DATASET/raw_images'),
        txts_dir=normpath('tmp/DATASET/raw_annotations'),
        dst_validatoion_dir=normpath('tmp/DATASET/yolo_validation'),
        dst_train_dir=normpath('tmp/DATASET/yolo_train'),
        validation_percentage=validation_split
    )
    # Change directory to train yolo
    os.makedirs('tmp/DATASET/yolo_data/images/train')
    os.makedirs('tmp/DATASET/yolo_data/images/val')
    os.makedirs('tmp/DATASET/yolo_data/labels/train')
    os.makedirs('tmp/DATASET/yolo_data/labels/val')

    movefiles(normpath('tmp/DATASET/yolo_validation/annotations/'),
              normpath('tmp/DATASET/yolo_data/labels/val/'))

    movefiles(normpath('tmp/DATASET/yolo_validation/images/'),
              normpath('tmp/DATASET/yolo_data/images/val/'))

    movefiles(normpath('tmp/DATASET/yolo_train/annotations/'),
              normpath('tmp/DATASET/yolo_data/labels/train/'))

    movefiles(normpath('tmp/DATASET/yolo_train/images/'),
              normpath('tmp/DATASET/yolo_data/images/train/'))

    d = {
        'train': os.path.abspath('tmp/DATASET/yolo_data/images/train'),
        'val': os.path.abspath('tmp/DATASET/yolo_data/images/val'),
        'nc': len(classes),
        'names': classes
    }
    dict_to_yaml(d, 'tmp/data.yaml')
    save_dir = normpath(req['save_dir'])
    # Training yolo
    train.run(data='tmp/data.yaml', imgsz=image_size, weights=weight,
              save_dir=save_dir, epochs=epochs, Log_url=Log_url, augment=augment)
    # delete temp file
    if os.path.exists('tmp') and os.path.isdir('tmp'):
        shutil.rmtree('tmp')

    resp_data = {
        'save_dir': save_dir,
        'model_dir': os.path.join(save_dir, 'weights/best.pt'),
        'classes': classes
    }
    # convert post_data to json
    resp_data = json.dumps(resp_data)
    # get end_url from os environment variable
    resp_url = os.environ.get('RESPONSE_URL')
    requests.post(resp_url, json=resp_data)
    return resp_data


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
