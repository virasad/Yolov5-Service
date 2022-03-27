# YOLOv5 train API

## How to use
```bash
python main_train.py
```
## Parameters:

```json
{
  "label":,
  "image_path":,
  "label_path":,
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
