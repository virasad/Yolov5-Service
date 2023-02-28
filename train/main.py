import json
import os
import requests
from fastapi import FastAPI

from preimutils.object_detection.yolo import AMRLImageAug
from preimutils.object_detection.yolo.coco2yolo import COCO2YOLO
from util.util import mkdir_p, movefiles
from util.coco2yolo import COCO2YOLO
from util.train_validation_sep import separate_test_val
from util.util import jsonfile2dict, remove_directory
from util.yaml_file import dict_to_yaml
from yolov7 import train
import uuid

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
                      weight: str = "yolov7_training.pt",
                      validation_split: float = 0.2,
                      data_type: str = "yolo",
                      save_dir: str = "results/",
                      task_id: str = "",
                      batch_size: int = 2,
                      response_url: str = None,
                      log_url: str = None,
                      classes: list = None,
                      except_url: str = None,
                      is_augment: bool = False,
                      count_of_each: int = 2,
                      from_scratch: bool = False,
                      ):
    
    try:
        if task_id == "":
            task_id = str(uuid.uuid4())
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
            '''if image_size != 640:
                return {
                    'message': 'image_size must be 640 for yolov5n.pt, yolov5s.pt, yolov5m.pt, yolov5l.pt, yolov5x.pt'}'''

        elif weight in weights_v6:
            '''if image_size != 1280:
                return {
                    'message': 'image_size must be 1280 for yolov5n6.pt, yolov5s6.pt, yolov5m6.pt, yolov5l6.pt, yolov5x6.pt'}'''
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
            dst_validatoion_dir=test_dir,
            dst_train_dir=train_dir,
            validation_percentage=validation_split
        )

        if is_augment:
            mkdir_p('/dataset/tmp')
            augment_base_dir = "/dataset/tmp"
            os.makedirs(normpath(f'{augment_base_dir}/yolo_train_augmented'))
            os.makedirs(normpath(f'{augment_base_dir}/yolo_validation_augmented'))
            # Augment train Data
            aug = AMRLImageAug(normpath(train_txts_dir),
                               normpath(train_images_dir),
                               normpath(f'{augment_base_dir}/yolo_train_augmented'))
            aug.auto_augmentation(count_of_each=count_of_each)

            # Augment validation Data
            aug = AMRLImageAug(normpath(validation_txts_dir),
                               normpath(validation_images_dir),
                               normpath(f'{augment_base_dir}/yolo_validation_augmented'))
            aug.auto_augmentation(count_of_each=count_of_each)

            # Change directory to train yolo
            os.makedirs(f'{augment_base_dir}/yolo_data/images/train')
            os.makedirs(f'{augment_base_dir}/yolo_data/images/val')
            os.makedirs(f'{augment_base_dir}/yolo_data/labels/train')
            os.makedirs(f'{augment_base_dir}/yolo_data/labels/val')

            movefiles(normpath(f'{augment_base_dir}/yolo_validation_augmented/annotations'),
                      normpath(f'{augment_base_dir}/yolo_data/labels/val'))
            movefiles(normpath(f'{augment_base_dir}/yolo_validation_augmented/images'),
                      normpath(f'{augment_base_dir}/yolo_data/images/val'))
            movefiles(normpath(f'{augment_base_dir}/yolo_train_augmented/annotations'),
                      normpath(f'{augment_base_dir}/yolo_data/labels/train'))
            movefiles(normpath(f'{augment_base_dir}/yolo_train_augmented/images'),
                      normpath(f'{augment_base_dir}/yolo_data/images/train'))

            train_images_dir = f'{augment_base_dir}/yolo_data/images/train'
            validation_images_dir = f'{augment_base_dir}/yolo_data/images/val'

        d = {
            'train': os.path.abspath(train_images_dir),
            'val': os.path.abspath(validation_images_dir),
            'nc': len(classes),
            'names': classes
        }
        data_yml = os.path.join(dataset_dir, 'data.yml')
        dict_to_yaml(d, data_yml)
        # save_dir = normpath(req['save_dir'])
        if not os.path.exists('yolov7_training.pt'):
            os.system("wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt")

        # Training yolo
        if from_scratch:
            cfg_type = weight.replace('.pt', '.yaml')
            train.run(data=data_yml, imgsz=image_size, weights='', cfg='yolov7/cfg/training/yolov7.yaml',
                      save_dir=save_dir, epochs=epochs, batch_size=batch_size,
                      hyp='yolov7/data/hyp.scratch.p5.yaml',
                      project=save_dir, name='', exists_ok=True, log_url=log_url,
                      response_url=response_url, task_id=task_id)
        else:
            train.run(data=data_yml, imgsz=image_size, weights='yolov7_training.pt', cfg='yolov7/cfg/training/yolov7.yaml',
                      save_dir=save_dir, epochs=epochs, batch_size=batch_size,
                      hyp='yolov7/data/hyp.scratch.custom.yaml',
                      project=save_dir, name='', exists_ok=True, log_url=log_url,
                      response_url=response_url, task_id=task_id)
        # # delete temp file
        remove_directory(augment_base_dir)
        # get end_url from os environment variable

        return {'message': 'training is done'}
    except Exception as e:
        print(str(e))
        remove_directory(augment_base_dir)
        if except_url:
            requests.post(url=except_url, data={'error message': e,
                                                'task_id': task_id}, timeout=2)
            return {'message': 'exception raised'}