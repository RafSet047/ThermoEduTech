from typing import Optional, List, Tuple, Dict
from tqdm import tqdm

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.modules.loss import _Loss

class ModelTrainer:
    def __init__(self, model: Module, train_loader: DataLoader, valid_loader: Optional[DataLoader] = None,
                 optimizer: Optional[Optimizer] = None, criterion: Optional[_Loss] = None,
                 schedulers: Optional[List[LRScheduler]] = None) -> None:
        self._model = model
        self._train_loader = train_loader
        self._valid_loader = valid_loader
        self._optimizer = optimizer
        self._criterion = criterion
        self._schedulers = schedulers

        self._best_weights: Dict = None

    @property
    def model(self) -> Module:
        return self._model

    @property
    def train_loader(self) -> DataLoader:
        return self._train_loader

    @property
    def valid_loader(self) -> Optional[DataLoader]:
        return self._valid_loader

    @property
    def optimizer(self) -> Optional[Optimizer]:
        return self._optimizer

    @property
    def criterion(self) -> Optional[_Loss]:
        return self._criterion

    @property
    def schedulers(self) -> Optional[List[LRScheduler]]:
        return self._schedulers

    def get_best_weights(self) -> Optional[Dict]:
        return self._best_weights
    
    def train_step(self) -> Tuple[float, float]:
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
        if self._valid_loader is None:
            print("Warning: No validation data is provided, skipping this step")
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

        best_loss = float('inf')
        train_avg_losses = []
        valid_avg_losses = []
        for epoch_idx in range(num_epochs):
            print("-----------------------------------")
            print("Epoch %d" % (epoch_idx+1))
            print("-----------------------------------")

            _, avg_train_loss = self.train_step()
            val_result = self.test_step()
            if not val_result is None:
                _, avg_val_loss = val_result
                for scheduler in self._schedulers:
                    scheduler.step(avg_val_loss)
                if avg_val_loss <= best_loss:
                    best_loss = avg_val_loss
                    self._best_weights = self._model.state_dict()
                valid_avg_losses.append(avg_val_loss)
            train_avg_losses.append(avg_train_loss) 
            print(f"Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
        return train_avg_losses, valid_avg_losses