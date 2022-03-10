import json
import os
import shutil

def dict_to_json(d, name):
    with open(name, 'w') as outfile:
        json.dump(d, outfile, ensure_ascii=False, indent=2)


def mkdir_p(dirname: str):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

def movefiles(src, dst):

    for file_name in os.listdir(src):
        # construct full file path
        source = os.path.join(src, file_name)
        destination = os.path.join(dst, file_name)
        # move only files
        if os.path.isfile(source):
            shutil.move(source, destination)

def jsonfile2dict(json_dir: str) -> dict:
    f = open(json_dir, 'r')
    d = json.load(f)
    f.close()
    return d