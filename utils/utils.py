import json
import yaml
from typing import *

def load_config(config_file: str) -> Dict[str, Any]:
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def write_config(config_file: str, data: Dict) -> None:
    with open(config_file, 'w') as file:
        yaml.safe_dump(data, file)
    file.close()

def write_json(config_file: str, data: Dict) -> None:
    with open(config_file, 'w') as file:
        json.dump(data, file)
    file.close()

def get_instance(module, name: str, config: Dict[str, Any], *args, **kwargs):
    cls = getattr(module, config[name]['type'])
    #constructor_params = cls.__init__.__code__.co_varnames
    instance_args = {**config[name]['args'], **kwargs}
    return cls(*args, **instance_args)
