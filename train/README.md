# YOLOv5 train API

## How to run
1. copy `train2.py` in yolov5 directory
2. run main file : `python main-train.py`

## Parameters:

```
{
"label":,
"imagePath":,
"labelPath" : , 
"log_url":,
"imageSize":,
"weight":,
"iaAugment":,
"augmentParams":,
"validationSplit":,
"dataType":"",
"save_dir":""
}
```

`imagePath` : Path of All images\
`imageSize` : Size of images train by yolo (default:640)\
`weights` : yolo weights names (`yolov5n.pt`, `yolov5s.pt`, ...)\
`isAugment` : if true , You have to set `augmentParams` for augmenting datasets\
`validationSplit` : train and validation split (default:0.2)\
`dataType` : `yolo` or `coco`\
`save_dir` : Path for saving model weights and results\
`labelPath` : Path of text files (for yolo only)

### COCO Dataset:
`label` : coco json annotation.

### YOLO Dataset:
`label` : list of clasees like `[class1, class2, ...]`
