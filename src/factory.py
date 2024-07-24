import os
import sys
import json
import shutil
import datetime
from typing import *

from loguru import logger

import torch.optim as optim
import torch.nn as nn

import models
import datasets
from shared_state import SharedState
from src.models.model import BaseModel
from src.dataset import ThermoDataset
from utils import get_instance, load_config

class Factory:
    def __init__(self, config_path: str) -> None:
        self._configs = load_config(config_path)
        self._shared_state = SharedState()

        self._output_dirpath = self._configs.get("output_dirpath", "")
        if '' == self._output_dirpath:
            logger.error(f"No given or empty `output_dirpath` in configs")
            raise ValueError("Wrong provided `output_dirpath`")
            
        if not os.path.exists('results'):
            os.makedirs('results')

        logger.info(f"Initialized the Factory from {config_path}")
        logger.info(json.dumps(self._configs, indent=2))

    def creat_output_dirpath(self, config_path: str):
        now = datetime.datetime.now()
        now = now.strftime("%d_%m_%Y_%H_%M_%S")
        self._output_dirpath = self._output_dirpath.split('/')[-1]
        self._output_dirpath = "_".join([self._output_dirpath, now])
        self._output_dirpath = os.path.join('results', self._output_dirpath)
        os.makedirs(self._output_dirpath, exist_ok=True)

        shutil.copyfile(config_path, os.path.join(self._output_dirpath, 'configs.yaml'))
        self._configs['output_dirpath'] = self._output_dirpath
        
    def create_model(self):
        model: BaseModel = get_instance(models, 'model', self._configs, state=self._shared_state)
        self._shared_state = model.get_shared_state()
        return model

    def create_dataset(self, subset: str):
        dataset: ThermoDataset = get_instance(datasets, subset, self._configs, shared_state=self._shared_state)
        self._shared_state = dataset.get_shared_state()
        return dataset

    def create_optimizer(self, model_parameters):
        optimizer = get_instance(optim, 'optimizer', self._configs, model_parameters)
        return optimizer
    
    def create_criterion(self):
        criterion = get_instance(nn, 'criterion', self._configs)
        return criterion

    def create_schedulers(self, optimizer):
        #FIXME
        scheduler = get_instance(optim.lr_scheduler, 'schedulers', self._configs, optimizer=optimizer)
        return [scheduler]
        #schedulers = []
        #for scheduler_config in self._configs.get('schedulers', []):
        #    scheduler = get_instance(optim.lr_scheduler, 'schedulers', scheduler_config, optimizer=optimizer)
        #    schedulers.append(scheduler)
        #return schedulers

    def get_batch_size(self) -> int:
        return self._configs.get("batch_size", 128)

    def get_device(self) -> str:
        return self._configs.get("device", 'cpu')

    def get_epochs(self) -> int:
        return self._configs.get("num_epochs", 3)

    def get_model_path(self) -> str:
        return os.path.join(self._output_dirpath, 'model.pt')
    
    def get_plots_save_path(self) -> str:
        return self._configs.get("plots_save_path", 'results.png')

    def get_output_dirpath(self) -> str:
        return self._output_dirpath

if __name__ == "__main__":
    c = sys.argv[1]

    f = Factory(c)
    train_data = f.create_dataset('train')
    valid_data = f.create_dataset('valid')
    model = f.create_model()

    optimizer = f.create_optimizer(model.parameters())
    criterion = f.create_criterion()
    schedulers = f.create_schedulers(optimizer)