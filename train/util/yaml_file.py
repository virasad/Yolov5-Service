import yaml

def dict_to_yaml(d, path):
    with open(path, 'w') as file:
        documents = yaml.dump(d, file)

def yaml_to_dict(path):
    with open(path) as file:
        d = yaml.load(file, Loader=yaml.FullLoader)

    return d