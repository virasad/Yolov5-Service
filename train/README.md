# YOLOv5 train API
At the first copy your dataset folder includes images and labels in volumes folder. So you should have a folder structure like this for yolo format dataset:

    volumes/
        dataset/
            images/
                *.jpg
            labels/
                *.txt
        weights/

## How to use
After 1) run train container with docker compose up command and 2) change train hyperparameters in req_train.py as you want, you can use the following commands to train your model.

```bash
python req_train.py
```
## HyperParameters:

```json
{
  "label":,
  "image_path":,
  "label_path":,
  "epochs":,
  "log_url":,
  "image_size":,
  "weight":,
  "is_augment":,
  "augment_params":,
  "validation_split":,
  "data_type":,
  "save_dir":
}
```

`image_path` : Path of All images\
`image_size` : Size of images train by yolo (default:640)\
`weights` : yolo weights names (`yolov5n.pt`, `yolov5s.pt`, ...)\
`is_augment` : if true , You have to set `augment_params` for augmenting datasets\
`validation_split` : train and validation split (default:0.2)\
`data_type` : `yolo` or `coco`\
`save_dir` : Path for saving model weights and results\
`label_path` : Path of text files (for yolo only)

### COCO Dataset:
`label` : coco json annotation.

### YOLO Dataset:
`label` : list of clasees like `[class1, class2, ...]`
