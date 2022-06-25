# Yolov5-Service

**Yolov5-Service** is a service that provides a RESTful API for Yolov5 models. run YOLOv5 with GPU.
**Easy As ABC**
## Inferencing
### How to run
#### Install Requirements
- Docker (https://docs.docker.com/engine/install/
- Docker Compose (https://docs.docker.com/compose/install/)
- Docker nvidia engine (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide)

#### Put weights
Put your pretrained weights in the folder (for example, `weights`)

#### Docker
Config docker compose file (`docker-compose.yml`)
```bash
docker compose up
```

### How to use
send sample image with `test_req.py` code.


## Training

### Send Train Request

```json
{
  "label": "label Path or Coco annotation json",
  "image_path": "images path",
  "weights_path": "path tp pretrained weights",
  "epochs" :"Number of train epochs"
  "is_augment": "true or false for data augmentation",
  "augment_params" : "Augmentation Params", -required if is_augment is True
  "validation_split": "Validation Split Number to split train and validation",
  "data_type": "Coco or yolo data type",  
  "image_size": "image size to resize",
  "log_url": "log url to get training status",
  "label_path": "label path to get label (used for yolo data type)",
  "save_dir": "save directory to save trained model"
}
```

