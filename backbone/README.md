# YOLOv5 inference API
At the first copy your pretrained weight (best.pt) in backbone/weights folder.So you should have a folder structure like this:

    backbone/
        weights/
            best.pt
        test_req.py
        detect.py
        app.py
        README.md

## How to use
After 1) run train container with docker compose up command and 2) change train hyperparameters as you like, you can use the following commands to inference  your model.

```bash
python test_req.py /path/to/image
```
With the above command, you get the rusults including bounding boxes and class names of the image.
