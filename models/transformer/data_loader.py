import sys
import yaml
import pandas as pd
from typing import *

import torch
from torch.utils.data import DataLoader, Dataset

class TransformerTabData(Dataset):
    def __init__(self, data_path: str, data_configs_path: str, device: str = 'cpu') -> None:
        super().__init__()
        self._df = None
        self._configs = None
        self.__device = device

        self.__load_data(data_path)
        self.__load_configs(data_configs_path)
        self.__parse_configs()
        self.__prepare_features()

    @property
    def df(self) -> Optional[pd.DataFrame]:
        return self._df.copy()

    @property
    def configs(self) -> Optional[pd.DataFrame]:
        return self._configs.copy()

    def __load_data(self, data_path: str) -> None:
        self._df = pd.read_csv(data_path)

    def __load_configs(self, configs_path: str) -> None:
        with open(configs_path, 'r') as y:
            self._configs: Dict = yaml.safe_load(y)
        y.close()
    
    def __parse_configs(self) -> None:
        self.__numerical_features: List[str] = self._configs['num_feats']
        self.__categorical_features: List[str] = self._configs['cat_feats']
        self.__target_feature: List[str] = self._configs['target_feature']

    def __prepare_features(self) -> None:
        self.__X_num = torch.as_tensor(self._df[self.__numerical_features].values, dtype=torch.float32, device=self.__device)
        self.__X_cat = torch.as_tensor(self._df[self.__categorical_features].values, dtype=torch.float32, device=self.__device)
        self.__Y = torch.as_tensor(self._df[self.__target_feature].values, dtype=torch.float32, device=self.__device)

    def __len__(self) -> int:
        return self._df.shape[0]

    def __getitem__(self, index) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return ((self.__X_num[index], self.__X_cat[index]), self.__Y[index])

if __name__ == "__main__":

    d_path = sys.argv[1]
    c_path = sys.argv[2]
    data = TransformerTabData(d_path, c_path)

    (num_sample, cat_sample), target = data[0]
    print(num_sample.shape)
    print(cat_sample.shape)
    print(target.shape)

