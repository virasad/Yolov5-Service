# Yolov5-Service

**Yolov5-Service** is a service that provides a RESTful API for Yolov5 models. run YOLOv5 with GPU.
**Easy As ABC**

# **Prerequisites**
- [Docker](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Docker nvidia engine](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide)
# **Inference**
Config docker compose file (`docker-compose.yml`) and run following commands:
```bash
docker compose up
```
With running service, you can access the service, two containers are created: one is for **inference** and another is for **training**.
## **How to use**
At the first time, you need to set a model path for inference.
1) For inference model with swagger gui which is available in fastapi by default, open http://127.0.0.1:8000/docs in your browser url. At the first, click on POST/inference button in **set-model** section, and then click on try it out button. After that you can set a path where your model is saved.after your model has been set, you get a success message. everything is ready to inference.

2) use **inference** service to inference a image. it is just like **set-model**, except that at this point you just select an image and see object detection result.

Also after running server and set a model, you can send POST request to server with following command in terminal:
  ```
    python test_req.py image_path
  ```
  where image_path is the path of your image.
    As an example, you can send request with following command
  ```sh
    python send_request.py ./images/image.jpg
  ```



# **Training**

If your docker service is running, you can send a train request to train a model. If not first run servers with following command:
```bash
docker-compose up
```
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

Everything is ready to start training. After training, you can use **inference** service to inference with new trained model.
Training is done by sending a POST request to **train** service with following command in terminal:
```bash
python train/req_train.py
```
Train's hyperparameters are set in `train/req_train.py`. in this file, you can change them as you like. Hyperparameters which are used in training are:
```json
{
  "label": "label Path or Coco annotation json",
  "image_path": "images path",
  "weights_path": "path to pretrained weights",
  "epochs" :"Number of train epochs",
  "is_augment": "true or false for data augmentation",
  "validation_split": "Validation Split Number to split train and validation",
  "data_type": "Coco or yolo data type",  
  "image_size": "image size to resize",
  "log_url": "log url to get training status",
  "label_path": "label path to get label (used for yolo data type)",
  "save_dir": "save directory to save trained model"
}
```

