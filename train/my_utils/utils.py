import json
import os


def dict_to_json(d, name):
    with open(name, 'w') as outfile:
        json.dump(d, outfile, ensure_ascii=False, indent=2)


def mkdir_p(dirname: str):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
