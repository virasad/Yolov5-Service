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
[ ] To be continued...

