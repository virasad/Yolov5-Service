import os
# from ..preimutils.preimutils.object_detection.yolo.validating_data import check_dataset
import shutil
from flask import Flask, request
from my_utils.utils import dict_to_json, mkdir_p
from my_utils.yaml_file import dict_to_yaml
from preimutils.preimutils.object_detection.yolo import AMRLImageAug
from preimutils.preimutils.object_detection.yolo.coco2yolo import COCO2YOLO
from preimutils.preimutils.object_detection.yolo.train_validation_sep import separate_test_val
from yolov5 import train2

app = Flask(__name__)
TRAIN_URL = ""

'''
{"modelPath":"",
 "isAugment":True,
 "weight":"",
 "validation_split":0.2,
 "augment_params":{"count_of_each":2},
 "log_url":"",
 "dataType":"COCO", # coco, yolo
 "imageSize":640}
 '''

@app.route(os.path.join(TRAIN_URL,'train'), methods=["POST"])
def train_model():
    path = request.form.get("modelPath")
    data_dict = request.form.get("modelPath")
    weight = input.weight
    iaAugment = request.form.get("iaAugment")
    validation_split = request.form.get("validation_split")
    dataType = request.form.get("dataType")
    image_size = request.form.get("imageSize")
    Log_url = request.form.get("log_url")

    # Create temp folder
    mkdir_p('tmp')
    # make raw Label Folder
    mkdir_p('tmp/DATASET/raw_annotations')


    if dataType == 'coco':
        # Save coco file in temp folder
        dict_to_json(data_dict, os.path.join('tmp', 'coco.json'))
        # Load coco.json file
        c2y = COCO2YOLO('tmp/coco.json', 'tmp/DATASET/raw_annotations')
        # Save Classes in coco_classes.txt
        c2y.save_classes('tmp/DATASET/')

        # Copy images to tmp
        mkdir_p('tmp/DATASET/raw_images')
        images_info = c2y._load_images_info()
        for images in images_info:
            name = images['file_name']
            shutil.copyfile(os.path.join(path, name), os.path.join('tmp/DATASET/raw_images', name))

        # Coco to YOLO
        c2y.coco2yolo()

        # Check Coco dataset
        # check_dataset('tmp/DATASET/raw_annotations', 'tmp/DATASET/raw_images')

        # Create Data Yaml file
        classes = c2y._categories()

    # Train test split
    separate_test_val(
        images_dir='tmp/DATASET/raw_images',
        txts_dir='tmp/DATASET/raw_annotations',
        dst_validatoion_dir='tmp/DATASET/yolo_validation',
        dst_train_dir='tmp/DATASET/yolo_train',
        validation_percentage=validation_split
    )

    if iaAugment:
        aug_params = request.form.get("augment_params")
        # Augment train Data
        aug = AMRLImageAug('tmp/DATASET/yolo_train/annotations',
                           'tmp/DATASET/yolo_train/images',
                           'tmp/DATASET/yolo_train_augmented')
        aug.auto_augmentation(**aug_params)

        # Augment validation Data
        aug = AMRLImageAug('tmp/DATASET/yolo_validation/annotations',
                           'tmp/DATASET/yolo_validation/images',
                           'tmp/DATASET/yolo_validation_augmented')
        aug.auto_augmentation(**aug_params)

        # Change directory to train yolo
        mkdir_p('tmp/DATASET/yolo_data/images/train')
        mkdir_p('tmp/DATASET/yolo_data/images/val')
        mkdir_p('tmp/DATASET/yolo_data/labels/train')
        mkdir_p('tmp/DATASET/yolo_data/labels/val')

        shutil.move('tmp/DATASET/yolo_validation_augmented/annotations', 'tmp/DATASET/yolo_data/labels/val')
        shutil.move('tmp/DATASET/yolo_validation_augmented/images', 'tmp/DATASET/yolo_data/images/val')
        shutil.move('tmp/DATASET/yolo_train_augmented/annotations', 'tmp/DATASET/yolo_data/labels/train')
        shutil.move('tmp/DATASET/yolo_train_augmented/images', 'tmp/DATASET/yolo_data/images/train')



    d = {
        'train': 'tmp/DATASET/yolo_data/images/train',
        'val': 'tmp/DATASET/yolo_data/images/val',
        'nc': len(classes),
        'names': list(classes.keys())
    }
    dict_to_yaml(d, 'tmp/data.yaml')
    save_dir = ""
    # Training yolo
    train2.run(data='tmp/data.yaml', imgsz=image_size, weights=weight, save_dir)

@app.route(os.path.join(TRAIN_URL,'test'), methods=["POST"])
def test():
    return request.form
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=os.environ.get("PORT", 5000))
