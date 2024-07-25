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
    """
    Instantiates an object from a specified module using a configuration dictionary.

    Args:
        module: The Python module containing the class to instantiate.
        name (str): The key in the config dictionary that specifies the object type and its initialization arguments.
        config (Dict[str, Any]): A dictionary containing the object configuration. 
            The dictionary should have the structure:
            {
                'object_name': {
                    'type': 'ClassName',
                    'args': {
                        # Arguments to pass to the class constructor
                    }
                }
            }
        *args: Additional positional arguments to pass to the class constructor.
        **kwargs: Additional keyword arguments to pass to the class constructor, overriding those in `config`.

    Returns:
        An instance of the specified class, initialized with the provided arguments.
    """
    cls = getattr(module, config[name]['type'])
    instance_args = {**config[name]['args'], **kwargs}
    return cls(*args, **instance_args)
