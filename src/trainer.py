from loguru import logger
from typing import Optional, List, Tuple, Dict
from tqdm import tqdm

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.modules.loss import _Loss

class ModelTrainer:
    """
    A class for training and evaluating PyTorch models.

    Attributes:
        model (Module): The PyTorch model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        valid_loader (Optional[DataLoader]): DataLoader for the validation data.
        optimizer (Optional[Optimizer]): Optimizer for updating model weights.
        criterion (Optional[_Loss]): Loss function used for training.
        schedulers (Optional[List[LRScheduler]]): Learning rate schedulers.
    """
    def __init__(self, model: Module, train_loader: DataLoader, valid_loader: Optional[DataLoader] = None,
                 optimizer: Optional[Optimizer] = None, criterion: Optional[_Loss] = None,
                 schedulers: Optional[List[LRScheduler]] = None) -> None:
        """
        Initializes the ModelTrainer with model, data loaders, optimizer, criterion, and schedulers.

        Args:
            model (Module): The PyTorch model to be trained.
            train_loader (DataLoader): DataLoader for the training data.
            valid_loader (Optional[DataLoader]): DataLoader for the validation data.
            optimizer (Optional[Optimizer]): Optimizer for updating model weights.
            criterion (Optional[_Loss]): Loss function used for training.
            schedulers (Optional[List[LRScheduler]]): Learning rate schedulers.
        """
        self._model = model
        self._train_loader = train_loader
        self._valid_loader = valid_loader
        self._optimizer = optimizer
        self._criterion = criterion
        self._schedulers = schedulers

        self._best_weights: Dict = None

    @property
    def model(self) -> Module:
        """Returns the model being trained."""
        return self._model

    @property
    def train_loader(self) -> DataLoader:
        """Returns the DataLoader for the training data."""
        return self._train_loader

    @property
    def valid_loader(self) -> Optional[DataLoader]:
        """Returns the DataLoader for the validation data, if provided."""
        return self._valid_loader

    @property
    def optimizer(self) -> Optional[Optimizer]:
        """Returns the optimizer used for training."""
        return self._optimizer

    @property
    def criterion(self) -> Optional[_Loss]:
        """Returns the loss function used for training."""
        return self._criterion

    @property
    def schedulers(self) -> Optional[List[LRScheduler]]:
        """Returns the list of learning rate schedulers, if provided."""
        return self._schedulers

    def get_best_weights(self) -> Optional[Dict]:
        """
        Returns the model weights with the best validation loss observed.

        Returns:
            Optional[Dict]: A dictionary of the model's state_dict with the best validation loss.
        """
        return self._best_weights
    
    def train_step(self) -> Tuple[float, float]:
        """
        Performs a single training step over the training dataset.

        Returns:
            Tuple[float, float]: Total loss and average loss for the epoch.
        """
        self._model.train()
        total_loss = 0.
        with tqdm(self._train_loader) as pbar:
            for X, y in pbar:
                y_pred = self._model(X)
                if len(y.size()) < 2:
                    y = y.reshape(-1, 1)
                
                self._optimizer.zero_grad()

                loss = self._criterion(y_pred, y)
                loss.backward()
                self._optimizer.step()

                total_loss += loss.item()
                pbar.set_description(f"Loss: {loss}")
        return total_loss, total_loss / len(self._train_loader)

    def test_step(self) -> Optional[Tuple[float, float]]:
        """
        Evaluates the model on the validation dataset.

        Returns:
            Optional[Tuple[float, float]]: Total loss and average loss for the validation set, or None if no validation data is provided.
        """
        if self._valid_loader is None:
            logger.warning("Warning: No validation data is provided, skipping this step")
            return
        self._model.eval()
        total_loss = 0.
        with torch.no_grad():
            for X, y in self._valid_loader:
                y_pred = self._model(X)
                if len(y.size()) < 2:
                    y = y.reshape(-1, 1)

                loss = self._criterion(y_pred, y)
                total_loss += loss.item()
        return total_loss, total_loss / len(self._valid_loader)

    def train(self, num_epochs: int) -> Tuple[List[float], List[float]]:
        """
        Trains the model for a specified number of epochs.

        Args:
            num_epochs (int): The number of epochs to train the model.

        Returns:
            Tuple[List[float], List[float]]: Lists of average training and validation losses per epoch.
        """

        best_loss = float('inf')
        train_avg_losses = []
        valid_avg_losses = []
        for epoch_idx in range(num_epochs):
            logger.info("-----------------------------------")
            logger.info("Epoch %d" % (epoch_idx+1))
            logger.info("-----------------------------------")

            _, avg_train_loss = self.train_step()
            val_result = self.test_step()
            if not val_result is None:
                _, avg_val_loss = val_result
                #for scheduler in self._schedulers:
                #    scheduler.step(avg_val_loss)
                if avg_val_loss <= best_loss:
                    best_loss = avg_val_loss
                    self._best_weights = self._model.state_dict()
                valid_avg_losses.append(avg_val_loss)
            train_avg_losses.append(avg_train_loss) 
            logger.info(f"Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
        return train_avg_losses, valid_avg_losses