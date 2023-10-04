import json
import os
import shutil

from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split


def dict_to_yaml(d, path):
    with open(path, "w") as file:
        documents = yaml.dump(d, file)
    return documents


def yaml_to_dict(path):
    with open(path) as file:
        d = yaml.load(file, Loader=yaml.FullLoader)
    return d


def yolo_train_test_split(
    image_dir: Path,
    label_dir: Path,
    dst_train_dir: Path,
    dst_test_dir: Path,
    train_test_ratio: list,
):
    image_paths = list(image_dir.glob("*"))
    images, labels = [], []
    for image_path in image_paths:
        label_basename = image_path.with_suffix(".txt").name
        label_path = label_dir / label_basename
        if label_path.exists():
            labels.append(label_path)
            images.append(image_path)
        else:
            print(f"{label_path} not exist")

    (
        train_image_paths,
        test_image_paths,
        train_label_paths,
        test_label_paths,
    ) = train_test_split(
        images,
        labels,
        train_size=train_test_ratio[0],
        test_size=train_test_ratio[1],
    )

    train_image_dir = dst_train_dir / "image"
    train_label_dir = dst_train_dir / "label"
    test_image_dir = dst_test_dir / "image"
    test_label_dir = dst_test_dir / "label"

    train_image_dir.mkdir(exist_ok=True, parents=True)
    train_label_dir.mkdir(exist_ok=True, parents=True)
    test_image_dir.mkdir(exist_ok=True, parents=True)
    test_label_dir.mkdir(exist_ok=True, parents=True)

    for image_path, label_path in zip(train_image_paths, train_label_paths):
        try:
            dst_image_path = train_image_dir / image_path.name
            dst_label_path = train_label_dir / label_path.name
            shutil.copy(image_path, dst_image_path)
            shutil.copy(label_path, dst_label_path)
        except Exception as e:
            print(str(e))

    for image_path, label_path in zip(test_image_paths, test_label_paths):
        try:
            dst_image_path = test_image_dir / image_path.name
            dst_label_path = test_label_dir / label_path.name
            shutil.copy(image_path, dst_image_path)
            shutil.copy(label_path, dst_label_path)
        except Exception as e:
            print(str(e))

    return (
        train_image_dir,
        train_label_dir,
        test_image_dir,
        test_label_dir,
    )


def dict_to_json(d, name):
    with open(name, "w") as outfile:
        json.dump(d, outfile, ensure_ascii=False, indent=2)


def movefiles(src, dst):
    for file_name in os.listdir(src):
        # construct full file path
        source = os.path.join(src, file_name)
        destination = os.path.join(dst, file_name)
        # move only files
        if os.path.isfile(source):
            shutil.move(source, destination)


def jsonfile2dict(json_dir: Path) -> dict:
    f = open(json_dir, "r")
    d = json.load(f)
    f.close()
    return d


def remove_directory(dir: str) -> None:
    if os.path.exists(dir) and os.path.isdir(dir):
        shutil.rmtree(dir)
