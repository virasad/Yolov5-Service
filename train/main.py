import json
import requests
from fastapi import FastAPI
import yaml
import urllib
import os

from util.coco2yolo import COCO2YOLO
from util.util import (
    jsonfile2dict,
    yolo_train_test_split,
    dict_to_yaml,
)
from yolov5 import train
import uuid
from pathlib import Path

import shutil


app = FastAPI(
    title="Yolov5 Train API",
)


@app.get("/")
def root():
    return {"message": "Welcome to the Train API get documentation at /docs"}


@app.post("/train")
async def train_model(
    label_file_str: str = "",
    image_dir_str: str = "",
    image_size: int = 640,
    epochs: int = 100,
    weight: str = "yolov5s.pt",
    train_test_ratio: list = [0.8, 0.2],
    data_type: str = "coco",
    save_dir: str = "/weights/",
    stream_id: str = "",
    task_id: str = "",
    batch_size: int = 16,
    response_url: str = "",
    log_url: str = "",
    info_url: str = "",
    except_url: str = "",
    use_augmentation: bool = False,
    classes: list = [],
    count_of_each: int = 2,
):
    try:
        image_dir = Path(image_dir_str).resolve()
        if task_id == "":
            task_id = str(uuid.uuid4())
        if data_type not in ["yolo", "coco"]:
            return {"message": "data_type must be yolo or coco"}
        # check images dir
        if not image_dir.is_dir():
            return {"message": "image_path is not a directory"}

        label_file = Path(label_file_str).resolve()
        dataset_dir = image_dir.parent
        if not label_file.is_file():
            if not label_file.suffix == ".json":
                return {"message": "label is not a json file"}
            return {"message": "label_path is not a file"}
        # check if label file is valid
        try:
            with open(label_file) as f:
                json.load(f)
        except json.JSONDecodeError:
            return {"message": "label is not a valid json file"}
        label_dir = dataset_dir / "labels"
        label_dir.mkdir(exist_ok=True)
        c2y = COCO2YOLO(jsonfile2dict(label_file), output=label_dir)
        c2y.coco2yolo()
        classes = c2y.get_classes()
        train_dir = dataset_dir / "train"
        train_dir.mkdir(exist_ok=True)
        test_dir = dataset_dir / "test"
        test_dir.mkdir(exist_ok=True)
        (
            train_image_dir,
            train_label_dir,
            val_image_dir,
            val_label_dir,
        ) = yolo_train_test_split(
            image_dir=image_dir,
            label_dir=label_dir,
            dst_train_dir=train_dir,
            dst_test_dir=test_dir,
            train_test_ratio=train_test_ratio,
        )
        data = {
            "train": str(train_image_dir.resolve()),
            "val": str(val_image_dir.resolve()),
            "nc": len(classes),
            "names": classes,
            "job_id": task_id,
        }

        last_weight = weight
        print("CHECKING FOR LAST WEIGHT")
        if last_data := json.loads(
            requests.get(
                info_url, json=json.dumps({"stream_id": stream_id})
            ).json()
        ):
            print("LAST WEIGHT FOUND")
            try:
                print(data["names"], last_data["names"])
                if data["names"] == last_data["names"]:
                    last_weight = f"{last_data['job_id']}.pt"
                    src_last_weight = (
                        f"http://{os.environ.get('LOCAL_DOMAIN')}/oms/"
                        / f"/oms/1/stream_{stream_id}/trains/{last_weight}"
                    )
                    print("ADDITIONAL TRAINING ", src_last_weight)
                    urllib.request.urlretrieve(src_last_weight, last_weight)
                else:
                    print("FROM SCRATCH")
            except Exception as e:
                print(e)
                print("FROM SCRATCH")
        data_yml_path = dataset_dir / "data.yaml"
        dict_to_yaml(data, data_yml_path)
        # Training yolo
        cfg_type = weight.replace(".pt", ".yaml")
        train.run(
            data=data_yml_path,
            imgsz=image_size,
            weights=last_weight,
            cfg=cfg_type,
            save_dir=save_dir,
            epochs=epochs,
            batch_size=batch_size,
            project=save_dir,
            name="",
            exists_ok=True,
            log_url=log_url,
            response_url=response_url,
            task_id=task_id,
        )

        return {"message": "training is done"}
    except Exception as e:
        if except_url:
            requests.post(
                url=except_url,
                data={"error message": e, "task_id": task_id},
                timeout=2,
            )
            return {"message": "exception raised"}
    finally:
        shutil.rmtree(train_dir)
        shutil.rmtree(test_dir)
        remove_pycache(".")
        if last_weight != "yolov5s.pt":
            os.remove(f"{last_data['job_id']}.pt")


def remove_pycache(path):
    for p in Path(path).rglob("__pycache__"):
        shutil.rmtree(p)
