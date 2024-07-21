import sys
from typing import *

import torch
from src.dataset import ThermoDataset
from src.shared_state import SharedState

import warnings
warnings.filterwarnings("ignore")

class ThermoTransformerDataset(ThermoDataset):
    def __init__(self, data_path: str, configs_path: str, shared_state: Optional[SharedState] = None, device: str = 'cpu') -> None:
        super().__init__(data_path, configs_path, shared_state, device)

        self.__parse_configs()
        self.__prepare_features()

        self._shared_state.num_categories = self._num_categories
        self._shared_state.num_continious = self._num_continious

    def __parse_configs(self) -> None:
        self.__numerical_features: List[str] = self._configs['num_feats']
        self._num_continious = len(self.__numerical_features)
        self.__categorical_features: List[str] = self._configs['cat_feats']
        self._num_categories = []
        self.__cat_features = []
        for cat_feat in self.__categorical_features:
            self._num_categories.append(cat_feat['num_uniques'])
            self.__cat_features.append(cat_feat['column_name'])

        self._num_categories = cast(tuple, self._num_categories)
        self.__target_feature: List[str] = self._configs['target_feature']

    def __prepare_features(self) -> None:
        self.__X_num = torch.as_tensor(self._df[self.__numerical_features].values, dtype=torch.float32, device=self._device)
        self.__X_cat = torch.as_tensor(self._df[self.__cat_features].values, dtype=torch.int32, device=self._device)
        self.__Y = torch.as_tensor(self._df[self.__target_feature].values, dtype=torch.float32, device=self._device)

    def __getitem__(self, index) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return ((self.__X_num[index], self.__X_cat[index]), self.__Y[index])

if __name__ == "__main__":

    d_path = sys.argv[1]
    c_path = sys.argv[2]

    s = SharedState()
    print(s.num_categories, s.num_continious)
    data = ThermoTransformerDataset(d_path, c_path, s)
    s = data.get_shared_state()
    print(s.num_categories, s.num_continious)

    (num_sample, cat_sample), target = data[0]
    print(num_sample.shape)
    print(cat_sample.shape)
    print(target.shape)

