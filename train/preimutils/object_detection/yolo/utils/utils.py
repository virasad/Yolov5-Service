import json
import os

def jsonfile2dict(json_dir: str) -> dict:
    f = open(json_dir, 'r')
    d = json.load(f)
    f.close()
    return d

def mkdir_p(dirname):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

def get_image_format(images_dir):
    img_name = os.listdir(images_dir)[0]

    image_type_list = ['.png', '.jpg', '.jpeg', '.bmp', '.dip', '.tif', '.tiff', '.jfif', '.pjpeg', '.pjp', '.webp']
    image_type_list_C = [t.upper() for t in image_type_list]

    for image_type in image_type_list:
        if image_type in img_name:
            return image_type
    
    for image_type in image_type_list:
        if image_type in img_name:
            return image_type