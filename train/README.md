# YOLOv5 Training API
For starting training, images and labels (yolo or coco format). So you should have a folder structure like this:

    inference/
        test_req.py
        .dockerignore
        Dockerfile
        detect.py
        main.py
        README.md
        requirements.txt
    train/
        util/
        yolov5/
        main.py
        README.md
        req_train.py
        Dockerfile
        requirements.txt
    volumes/
        dataset/
            images/
              **image01.jpg**
              **image02.jpg**
              ...
            labels/
              **coco_annotation.json** >> (COCO Format)
              image01.txt(YOLO Format)
              image02.txt(YOLO Format)
              ...(YOLO Format)
        weights/
    .gitignore
    docker-compose.yml
    README.md

## HyperParameters

Train's hyperparameters are set in `train/req_train.py`. in this file, you can change them as you like. Hyperparameters which are used in training are:
```json
{
  "label": "label Path or Coco annotation json",
  "image_path": "images path",
  "weights_path": "path to pretrained weights",
  "epochs" :"Number of train epochs",
  "is_augment": "true or false for data augmentation",
  "validation_split": "Validation Split rate to split train and validation",
  "data_type": "COCO or YOLO data format",  
  "image_size": "image size to resize",
  "log_url": "log url to get training status",
  "label_path": "label path to get label (used for yolo data type)",
  "save_dir": "save directory to save trained model"
}
```
