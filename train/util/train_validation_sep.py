import random
import os
import glob
import shutil
from tqdm import tqdm
import pandas
from sklearn.model_selection import train_test_split


def separate_test_val(images_dir, txts_dir, dst_validatoion_dir, dst_train_dir, validation_percentage=0.2):
    """
    Seperating Train and validation to their related directories
    Args:
        txts_dir: all xmls files directory.
        images_dir: your images directory.
        dst_validatoion_dir:destination directory to save validations images and xmls after seperating
        dst_train_dir:destination directory to save train images and xmls after seperating
    Returns:
        No return
    """
    images_p = list(glob.glob(os.path.join(images_dir, '*')))
    txts_p = list(glob.glob(os.path.join(txts_dir, '*.txt')))

    images = []
    txts = []
    for image in images_p:
        image_basename = os.path.basename(image)
        image_name, image_extension = os.path.splitext(image_basename)
        try:
            glob_txt = list(glob.glob(os.path.join(txts_dir, image_name + '*.txt')))[0]
            txts.append(glob_txt)
            images.append(image)
        except IndexError:
            print('{} not exist'.format(image_name + '*.txt'))

    # separate validation and train
    validation_max_index = int(validation_percentage * len(images))
    validation_images = images[:validation_max_index]
    validation_txts = txts[:validation_max_index]
    train_images = images[validation_max_index:]
    train_txts = txts[validation_max_index:]

    # copy validation images and txts
    train_images_dir = os.path.join(dst_train_dir, 'images')
    train_txts_dir = os.path.join(dst_train_dir, 'labels')
    validation_images_dir = os.path.join(dst_validatoion_dir, 'images')
    validation_txts_dir = os.path.join(dst_validatoion_dir, 'labels')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_txts_dir, exist_ok=True)
    os.makedirs(validation_images_dir, exist_ok=True)
    os.makedirs(validation_txts_dir, exist_ok=True)

    # move validation images and txts

    for idx, image in enumerate(validation_images):
        image_basename = os.path.basename(image)
        image_name, image_extension = os.path.splitext(image_basename)
        txt = validation_txts[idx]
        shutil.move(image, os.path.join(validation_images_dir, image_basename))
        shutil.move(txt, os.path.join(validation_txts_dir, os.path.basename(txt)))

    # move train images and txts
    for idx, image in enumerate(train_images):
        image_basename = os.path.basename(image)
        image_name, image_extension = os.path.splitext(image_basename)
        txt = train_txts[idx]
        shutil.move(image, os.path.join(train_images_dir, image_basename))
        shutil.move(txt, os.path.join(train_txts_dir, os.path.basename(txt)))

    print('Done')
    return train_images_dir, train_txts_dir, validation_images_dir, validation_txts_dir
