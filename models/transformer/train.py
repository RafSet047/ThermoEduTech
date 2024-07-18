import os
import sys
import yaml
from typing import Dict, Tuple
from tqdm import tqdm

from data_loader import TransformerTabData
from ft_transformer import FTTransformer

import torch
from torch.utils.data import DataLoader

#utils
def load_configs(path: str) -> Dict:
    with open(path, 'r') as y:
        configs = yaml.safe_load(y)
    y.close()
    return configs

class TrainTransformer:
    def __init__(self, data_dirpath: str, data_configs_path: str, model_configs_path: str) -> None:

        self._model_configs = load_configs(model_configs_path)
        self.__setup_datasets(data_dirpath, data_configs_path)
        self.__setup_model()

    def __setup_datasets(self, data_dirpath: str, data_configs_path: str) -> None:
        train_data = TransformerTabData(data_path=os.path.join(data_dirpath, 'train.csv'),
                                        data_configs_path=data_configs_path,
                                        device='cuda')
        self._train_loader = DataLoader(dataset=train_data, 
                                        batch_size=self._model_configs['batch_size'],
                                        shuffle=True)

        valid_data = TransformerTabData(data_path=os.path.join(data_dirpath, 'valid.csv'),
                                        data_configs_path=data_configs_path,
                                        device='cuda')
        self._valid_loader = DataLoader(dataset=valid_data, 
                                        batch_size=self._model_configs['batch_size'],
                                        shuffle=False)
        self._num_categories = train_data.num_categories
        self._num_continious = train_data.num_continious

    def __setup_model(self) -> None:
        self._model = FTTransformer(categories=self._num_categories,
                            num_continuous=self._num_continious,
                            heads=8, dim=32, dim_out=1, depth=6,
                            attn_dropout=0.2, ff_dropout=0.2,
                            device='cuda'
                            )
        self._criterion = torch.nn.MSELoss()
        self._optimizer = torch.optim.Adam(params=self._model.parameters(),
                                           lr=1e-4)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer)

    def train_step(self) -> Tuple[float, float]:
        self._model.train()
        total_loss = 0.
        for (x_num, x_cat), y in tqdm(self._train_loader):
            pred = self._model(x_cat, x_num)
            y = y.reshape(-1, 1)

            self._optimizer.zero_grad()

            loss = self._criterion(pred, y)
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()
        return total_loss, total_loss / len(self._train_loader)

    def eval_step(self) -> Tuple[float, float]:
        self._model.eval()
        total_loss = 0.
        with torch.no_grad():
            for (x_num, x_cat), y in self._valid_loader:
                pred = self._model(x_cat, x_num)
                y = y.reshape(-1, 1)

                loss = self._criterion(pred, y)
                total_loss += loss.item()
        avg_loss = total_loss / len(self._valid_loader)
        return total_loss, avg_loss

    def train(self):

        for epoch_idx in range(self._model_configs['epochs']):
            print("-----------------------------------")
            print("Epoch %d" % (epoch_idx+1))
            print("-----------------------------------")

            train_loss, avg_train_loss = self.train_step()
            self._scheduler.step(train_loss)

            val_loss, avg_val_loss = self.eval_step()
            print(f"Train Loss: {train_loss}, Validation Loss: {val_loss}")
        
        torch.save(self._model.state_dict(), self._model_configs['model_path'])
        return
            
if __name__ == "__main__":
    d_path = sys.argv[1]
    c_path = sys.argv[2]
    m_path = sys.argv[3]
    
    trainer = TrainTransformer(d_path, c_path, m_path)
    trainer.train()
    

    

    

