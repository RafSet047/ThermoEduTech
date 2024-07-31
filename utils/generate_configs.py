import os
import sys
from utils import write_config

from itertools import product

def get_default_configs(data_dirpath, data_type = "TimeSeriesTransformerDataset"):
    configs = {}
    configs_path = f'{data_dirpath}/configs.yaml'
    device = 'cpu'
    for sub in ['train', 'valid', 'test']:
        configs[sub] = {
            'type' : data_type,
            'args' : {
                'data_path' : f'{data_dirpath}/{sub}.csv',
                'configs_path' : configs_path,
                'device' : device
            }
        }
    configs['schedulers'] = {
        'type': 'ReduceLROnPlateau',
        'args': {}
    } 
    configs['device'] = device
    configs['num_epochs'] = 50
    return configs

def generate_configs(output_dirpath):
    os.makedirs(output_dirpath, exist_ok=True)
    
    models_datas = [
        ("TabTransformer", "ThermoTransformerDataset"),
        ("FTabTransformer", "ThermoTransformerDataset"),
        ("STabTransformer", "ThermoTransformerDataset"),
        ("VanillaTransformer", "TimeSeriesTransformerDataset"),
        ("Informer", "TimeSeriesTransformerDataset")
    ]
    
    data_dirpaths = ['dataset']
    losses = ['HuberLoss'] 
    optimizers = ['Adam']
    learning_rates = [0.001]
    batch_sizes = [256]
    
    combinations = list(product(data_dirpaths, models_datas, losses, optimizers, learning_rates, batch_sizes))
    for i, combination in enumerate(combinations):
        dirpath, model_data, loss, optimizer, lr, bs = combination
        model, data = model_data
        configs = get_default_configs(dirpath, data)
        configs['model'] = {
            'type' : model,
            'args': {
                'configs_path': "src/models/transformer/configs.yaml",
                'device': 'cpu'
            }
        }
        configs['criterion'] = {
            'type': loss,
            'args' : {}
        }
        configs['optimizer'] = {
            'type': optimizer,
            'args' : {
                'lr' : lr
            }
        }
        configs['batch_size'] = bs
        configs['output_dirpath'] = os.path.join('results', model)

        write_config(os.path.join(output_dirpath, f"configs_{i}.yaml"), configs)

if __name__ == "__main__":
    generate_configs(sys.argv[1])