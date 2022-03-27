import os
import shutil
from flask import Flask, request
from my_utils.utils import dict_to_json, mkdir_p, movefiles, jsonfile2dict
from my_utils.yaml_file import dict_to_yaml
from preimutils.preimutils.object_detection.yolo import AMRLImageAug
from preimutils.preimutils.object_detection.yolo.coco2yolo import COCO2YOLO
from preimutils.preimutils.object_detection.yolo.train_validation_sep import separate_test_val
from yolov5 import train

normpth = os.path.normpath

app = Flask(__name__)
TRAIN_URL = "/"


@app.route(os.path.join(TRAIN_URL, 'train'), methods=["POST"])
def train_model():
    req = request.json
    data = req["label"]
    image_path = req["imagePath"]
    weight = req["weight"]
    iaAugment = req["iaAugment"]
    validation_split = req["validationSplit"]
    dataType = req["dataType"]

    try:
        image_size = req["imageSize"]
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
    image_path = normpth(image_path)

    if dataType == "coco":
        # Save coco file in temp folder
        dict_to_json(data, os.path.join('tmp', 'coco.json'))
        # Load coco.json file
        c2y = COCO2YOLO(jsonfile2dict(normpth('tmp/coco.json')), normpth('tmp/DATASET/raw_annotations'))
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

    elif dataType == 'yolo':
        label_path = req["labelPath"]
        classes = data
        os.makedirs('tmp/DATASET/raw_images')
        print(image_path, label_path)
        for name in os.listdir(image_path):
            shutil.copyfile(os.path.join(image_path, name), os.path.join(normpth('tmp/DATASET/raw_images'), name))

        for name in os.listdir(normpth(label_path)):
            shutil.copyfile(os.path.join(label_path, name), os.path.join(normpth('tmp/DATASET/raw_annotations'), name))

    # Train test split
    separate_test_val(
        images_dir=normpth('tmp/DATASET/raw_images'),
        txts_dir=normpth('tmp/DATASET/raw_annotations'),
        dst_validatoion_dir=normpth('tmp/DATASET/yolo_validation'),
        dst_train_dir=normpth('tmp/DATASET/yolo_train'),
        validation_percentage=validation_split
    )

    if iaAugment:
        aug_params = req["augmentParams"]
        # Augment train Data
        aug = AMRLImageAug(normpth('tmp/DATASET/yolo_train/annotations'),
                           normpth('tmp/DATASET/yolo_train/images'),
                           normpth('tmp/DATASET/yolo_train_augmented'))
        aug.auto_augmentation(**aug_params)

        # Augment validation Data
        aug = AMRLImageAug(normpth('tmp/DATASET/yolo_validation/annotations'),
                           normpth('tmp/DATASET/yolo_validation/images'),
                           normpth('tmp/DATASET/yolo_validation_augmented'))
        aug.auto_augmentation(**aug_params)

        # Change directory to train yolo
        os.makedirs('tmp/DATASET/yolo_data/images/train')
        os.makedirs('tmp/DATASET/yolo_data/images/val')
        os.makedirs('tmp/DATASET/yolo_data/labels/train')
        os.makedirs('tmp/DATASET/yolo_data/labels/val')

        movefiles(normpth('tmp/DATASET/yolo_validation_augmented/annotations'),
                  normpth('tmp/DATASET/yolo_data/labels/val'))
        movefiles(normpth('tmp/DATASET/yolo_validation_augmented/images'), normpth('tmp/DATASET/yolo_data/images/val'))
        movefiles(normpth('tmp/DATASET/yolo_train_augmented/annotations'),
                  normpth('tmp/DATASET/yolo_data/labels/train'))
        movefiles(normpth('tmp/DATASET/yolo_train_augmented/images'), normpth('tmp/DATASET/yolo_data/images/train'))

    else:
        # Change directory to train yolo
        os.makedirs('tmp/DATASET/yolo_data/images/train')
        os.makedirs('tmp/DATASET/yolo_data/images/val')
        os.makedirs('tmp/DATASET/yolo_data/labels/train')
        os.makedirs('tmp/DATASET/yolo_data/labels/val')

        movefiles(normpth('tmp/DATASET/yolo_validation/annotations/'),
                  normpth('tmp/DATASET/yolo_data/labels/val/'))

        movefiles(normpth('tmp/DATASET/yolo_validation/images/'),
                  normpth('tmp/DATASET/yolo_data/images/val/'))

        movefiles(normpth('tmp/DATASET/yolo_train/annotations/'),
                  normpth('tmp/DATASET/yolo_data/labels/train/'))

        movefiles(normpth('tmp/DATASET/yolo_train/images/'),
                  normpth('tmp/DATASET/yolo_data/images/train/'))

    d = {
        'train': os.path.abspath('tmp/DATASET/yolo_data/images/train'),
        'val': os.path.abspath('tmp/DATASET/yolo_data/images/val'),
        'nc': len(classes),
        'names': classes
    }
    dict_to_yaml(d, 'tmp/data.yaml')
    save_dir = normpth(req['save_dir'])
    # Training yolo
    train.run(data='tmp/data.yaml', imgsz=image_size, weights=weight, save_dir=save_dir, log_url=Log_url)

    # delete temp file
    if os.path.exists('tmp') and os.path.isdir('tmp'):
        shutil.rmtree('tmp')


if __name__ == "__main__":
    app.run(debug=True, port=8000)
