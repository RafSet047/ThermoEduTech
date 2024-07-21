import sys
from typing import *

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
        return self._configs.get("epochs", 10)

    def get_model_path(self) -> str:
        return self._configs.get("model_path", "model.pt")

if __name__ == "__main__":
    c = sys.argv[1]

    f = Factory(c)
    train_data = f.create_dataset('train')
    valid_data = f.create_dataset('valid')
    model = f.create_model()

    optimizer = f.create_optimizer(model.parameters())
    criterion = f.create_criterion()
    schedulers = f.create_schedulers(optimizer)