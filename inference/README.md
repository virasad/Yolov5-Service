# YOLOv5 inference API
At the first it's necessary to load a model for inference.copy your pretrained model to the volumes/weights folder.

So you should have a folder structure like this:

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
            labels/
        weights/
            **your_model.pt**
    .gitignore
    docker-compose.yml
    README.md


## How to use
After 1) run train container with docker compose up command and 2) change train hyperparameters as you like, you can use the following commands to inference  your model.

```bash
python test_req.py /path/to/image
```
With the above command, you get the rusults including bounding boxes and class names of the image.
