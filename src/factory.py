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
from utils import get_instance, load_config, write_config

class Factory:
    """
    A factory class for creating various components such as models, datasets, optimizers,
    and criteria from a configuration file. It also manages shared states and directories
    for outputs and results.

    Attributes:
        _configs (Dict[str, Any]): Configuration parameters loaded from the provided config path.
        _shared_state (SharedState): An object for sharing state between different components.
        _output_dirpath (str): The directory path where outputs and results are saved.

    Methods:
        creat_output_dirpath(config_path: str): Creates the output directory path based on the current time and configuration.
        create_model(): Creates and returns a model instance using the configuration.
        create_dataset(subset: str): Creates and returns a dataset instance for the specified subset.
        create_optimizer(model_parameters): Creates and returns an optimizer for the given model parameters.
        create_criterion(): Creates and returns the loss criterion.
        create_schedulers(optimizer): Creates and returns a list of learning rate schedulers for the optimizer.
        get_batch_size() -> int: Returns the batch size for training.
        get_device() -> str: Returns the device type (CPU or GPU) for computation.
        get_epochs() -> int: Returns the number of training epochs.
        get_model_path() -> str: Returns the path where the trained model is saved.
        get_plots_save_path() -> str: Returns the path where plots are saved.
        get_output_dirpath() -> str: Returns the current output directory path.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initializes the Factory with a configuration file.

        Args:
            config_path (str): Path to the configuration file.

        Raises:
            ValueError: If the output directory path is not provided or is empty.
        """
        self._configs = load_config(config_path)
        self._shared_state = SharedState()

        self._output_dirpath = self._configs.get("output_dirpath", "")
        if '' == self._output_dirpath:
            logger.error(f"No given or empty `output_dirpath` in configs")
            raise ValueError(self._output_dirpath)
            
        if not os.path.exists('results'):
            os.makedirs('results')

        logger.info(f"Initialized the Factory from {config_path}")
        logger.info(json.dumps(self._configs, indent=2))

    def create_output_dirpath(self):
        """
        Creates the output directory path based on the current time and configuration.

        Args:
            config_path (str): The path to the configuration file to be copied into the output directory.
        """
        now = datetime.datetime.now()
        now = now.strftime("%d_%m_%Y_%H_%M_%S")
        self._output_dirpath = self._output_dirpath.split('/')[-1]
        self._output_dirpath = "_".join([self._output_dirpath, now])
        self._output_dirpath = os.path.join('results', self._output_dirpath)
        os.makedirs(self._output_dirpath, exist_ok=True)

        self._configs['output_dirpath'] = self._output_dirpath
        write_config(os.path.join(self._output_dirpath, 'configs.yaml'), self._configs)

        shutil.copyfile(self._configs['model']['args']['configs_path'], os.path.join(self._output_dirpath, 'model_configs.yaml'))
        shutil.copyfile(self._configs['train']['args']['configs_path'], os.path.join(self._output_dirpath, 'data_configs.yaml'))
        
    def create_model(self) -> BaseModel:
        """
        Creates and returns a model instance using the configuration.

        Returns:
            BaseModel: The instantiated model.
        """
        model: BaseModel = get_instance(models, 'model', self._configs, state=self._shared_state)
        self._shared_state = model.get_shared_state()
        return model

    def create_dataset(self, subset: str) -> ThermoDataset:
        """
        Creates and returns a dataset instance for the specified subset.

        Args:
            subset (str): The name of the dataset subset (e.g., 'train', 'valid').

        Returns:
            ThermoDataset: The instantiated dataset.
        """
        dataset: ThermoDataset = get_instance(datasets, subset, self._configs, shared_state=self._shared_state)
        self._shared_state = dataset.get_shared_state()
        return dataset

    def create_optimizer(self, model_parameters):
        """
        Creates and returns an optimizer for the given model parameters.

        Args:
            model_parameters: The parameters of the model to be optimized.

        Returns:
            Optimizer: The instantiated optimizer.
        """
        optimizer = get_instance(optim, 'optimizer', self._configs, model_parameters)
        return optimizer
    
    def create_criterion(self):
        """
        Creates and returns the loss criterion.

        Returns:
            _Loss: The instantiated loss criterion.
        """
        criterion = get_instance(nn, 'criterion', self._configs)
        return criterion

    def create_schedulers(self, optimizer):
        """
        Creates and returns a list of learning rate schedulers for the optimizer.

        Args:
            optimizer: The optimizer for which to create the schedulers.

        Returns:
            List[LRScheduler]: A list of instantiated learning rate schedulers.
        """
        #FIXME
        scheduler = get_instance(optim.lr_scheduler, 'schedulers', self._configs, optimizer=optimizer)
        return [scheduler]
        #schedulers = []
        #for scheduler_config in self._configs.get('schedulers', []):
        #    scheduler = get_instance(optim.lr_scheduler, 'schedulers', scheduler_config, optimizer=optimizer)
        #    schedulers.append(scheduler)
        #return schedulers

    def get_batch_size(self) -> int:
        """
        Returns the batch size for training.

        Returns:
            int: The batch size.
        """
        return self._configs.get("batch_size", 128)

    def get_device(self) -> str:
        """
        Returns the device type (CPU or GPU) for computation.

        Returns:
            str: The device type.
        """
        return self._configs.get("device", 'cpu')

    def get_epochs(self) -> int:
        """
        Returns the number of training epochs.

        Returns:
            int: The number of epochs.
        """
        return self._configs.get("num_epochs", 3)

    def get_model_path(self) -> str:
        """
        Returns the path where the trained model is saved.

        Returns:
            str: The model save path.
        """
        return os.path.join(self._output_dirpath, 'model.pt')
    
    def get_plots_save_path(self) -> str:
        """
        Returns the path where plots are saved.

        Returns:
            str: The plots save path.
        """
        return self._configs.get("plots_save_path", 'results.png')

    def get_output_dirpath(self) -> str:
        """
        Returns the current output directory path.

        Returns:
            str: The output directory path.
        """
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