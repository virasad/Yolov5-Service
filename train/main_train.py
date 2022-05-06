import os
import shutil
from flask import Flask, request
from utils.utils import dict_to_json, mkdir_p, movefiles, jsonfile2dict
import json
from utils.yaml_file import dict_to_yaml
from preimutils.object_detection.yolo import AMRLImageAug
from preimutils.object_detection.yolo.coco2yolo import COCO2YOLO
from preimutils.object_detection.yolo.train_validation_sep import separate_test_val
from yolov5 import train
import requests

normpath = os.path.normpath

app = Flask(__name__)
TRAIN_URL = "/"


@app.route(os.path.join(TRAIN_URL, 'train'), methods=["POST"])
def train_model():
    req = request.json
    data = req["label"]
    image_path = req["image_path"]
    weight = req["weight"]
    is_augment = req["is_augment"]
    validation_split = req["validation_split"]
    data_type = req["data_type"]
    pre_trained_path = req.get("pretrained_path", '')

    try:
        image_size = req["image_size"]
    except:
        image_size = 640

    try:
        Log_url = req["log_url"]
    except:
        Log_url = None

    # Create temp folder
    mkdir_p('tmp')
    # make raw Label Folder
    os.makedirs(os.path.join('tmp/DATASET/raw_annotations/'))
    image_path = normpath(image_path)

    if data_type == "coco":
        # Save coco file in temp folder
        dict_to_json(data, os.path.join('tmp', 'coco.json'))
        # Load coco.json file
        c2y = COCO2YOLO(jsonfile2dict(normpath('tmp/coco.json')), normpath('tmp/DATASET/raw_annotations'))
        # Save Classes in coco_classes.txt
        c2y.save_classes(os.path.join('tmp/DATASET/'))

        # Copy images to tmp
        os.makedirs('tmp/DATASET/raw_images')
        images_info = c2y._load_images_info()
        for images in images_info:
            name = images_info[images][0]
            shutil.copyfile(os.path.join(image_path, name), os.path.join('tmp/DATASET/raw_images', name))

        # Coco to YOLO
        c2y.coco2yolo()

        # Check Coco dataset
        # check_dataset(normpth('tmp/DATASET/raw_annotations'), normpth('tmp/DATASET/raw_images'))

        # Create Data Yaml file
        classes = c2y._categories()
        classes = list(classes.values())

    elif data_type == 'yolo':
        label_path = req["label_path"]
        classes = data
        os.makedirs('tmp/DATASET/raw_images')
        print(image_path, label_path)
        for name in os.listdir(image_path):
            shutil.copyfile(os.path.join(image_path, name), os.path.join(normpath('tmp/DATASET/raw_images'), name))

        for name in os.listdir(normpath(label_path)):
            shutil.copyfile(os.path.join(label_path, name), os.path.join(normpath('tmp/DATASET/raw_annotations'), name))

    # Train test split
    separate_test_val(
        images_dir=normpath('tmp/DATASET/raw_images'),
        txts_dir=normpath('tmp/DATASET/raw_annotations'),
        dst_validatoion_dir=normpath('tmp/DATASET/yolo_validation'),
        dst_train_dir=normpath('tmp/DATASET/yolo_train'),
        validation_percentage=validation_split
    )

    if is_augment:
        aug_params = req["augment_params"]
        # Augment train Data
        aug = AMRLImageAug(normpath('tmp/DATASET/yolo_train/annotations'),
                           normpath('tmp/DATASET/yolo_train/images'),
                           normpath('tmp/DATASET/yolo_train_augmented'))
        aug.auto_augmentation(**aug_params)

        # Augment validation Data
        aug = AMRLImageAug(normpath('tmp/DATASET/yolo_validation/annotations'),
                           normpath('tmp/DATASET/yolo_validation/images'),
                           normpath('tmp/DATASET/yolo_validation_augmented'))
        aug.auto_augmentation(**aug_params)

        # Change directory to train yolo
        os.makedirs('tmp/DATASET/yolo_data/images/train')
        os.makedirs('tmp/DATASET/yolo_data/images/val')
        os.makedirs('tmp/DATASET/yolo_data/labels/train')
        os.makedirs('tmp/DATASET/yolo_data/labels/val')

        movefiles(normpath('tmp/DATASET/yolo_validation_augmented/annotations'),
                  normpath('tmp/DATASET/yolo_data/labels/val'))
        movefiles(normpath('tmp/DATASET/yolo_validation_augmented/images'), normpath('tmp/DATASET/yolo_data/images/val'))
        movefiles(normpath('tmp/DATASET/yolo_train_augmented/annotations'),
                  normpath('tmp/DATASET/yolo_data/labels/train'))
        movefiles(normpath('tmp/DATASET/yolo_train_augmented/images'), normpath('tmp/DATASET/yolo_data/images/train'))

    else:
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
    train.run(data='tmp/data.yaml', imgsz=image_size, weights=weight, save_dir=save_dir, log_url=Log_url, weights=pre_trained_path)

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

        


if __name__ == "__main__":
    app.run(debug=True, port=8000)
