import sys
from typing import *
import random
import torch
from src.dataset import ThermoDataset
from src.shared_state import SharedState
from src.datasets.time_series_augmentation import Augmentation

import warnings
warnings.filterwarnings("ignore")

STEP_SIZE_DEFAULT = 6

class TimeSeriesTransformerDataset(ThermoDataset):
    def __init__(self, data_path: str, configs_path: str, shared_state: Optional[SharedState] = None, augment: bool = False, device: str = 'cpu') -> None:
        super().__init__(data_path, configs_path, shared_state, device)

        self.__parse_configs() 

        shared_state.num_features = len(self.__columns)
        shared_state.sequence_length = self.__step_size

        self.augment = Augmentation() if augment else None

    def __parse_configs(self) -> None:
        """
        Parses the configuration file to extract numerical and categorical features,
        as well as the target feature.
        """
        self.__numerical_features: List[str] = self._configs['num_feats']
        self.__categorical_features: List[str] = self._configs['cat_feats']
        self.__cat_features = []
        for cat_feat in self.__categorical_features:
            self.__cat_features.append(cat_feat['column_name'])

        self.__target_feature: List[str] = self._configs['target_feature']
        self.__columns = self.__numerical_features + self.__cat_features

        self.__step_size = self._configs.get("step_size", STEP_SIZE_DEFAULT)

    def prepare_data(self):
        
        x = self._df[self.__columns]
        y = self._df[self.__target_feature].values
        N = len(x)
        
        _x = []
        for i in range(N - self.__step_size):
            inp = []
            for col in self.__columns:
                values = torch.FloatTensor(x[col].iloc[i:(i + self.__step_size)].values).view(self.__step_size, 1)
                inp.append(values)
            inp = torch.cat(inp, dim=1)
            _x.append(inp)
        
        self._x = torch.stack(_x)  # Shape: (N - step_size, step_size, F)
        self._y = torch.FloatTensor([y[i + self.__step_size] for i in range(N - self.__step_size)])

    def __len__(self) -> int:
        return len(self._df) - self.__step_size

    def get_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._x, self._y

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._df.loc[index:index + self.__step_size - 1, self.__columns].values
        y = self._df.loc[index + self.__step_size, self.__target_feature]
        x = torch.FloatTensor(x)
        y = torch.FloatTensor([y])
        if self.augment is None:
            return x, y
        do_augment = random.choices([True, False], weights=[0.8, 0.2], k=1)[0]
        if do_augment:
            x = self.augment.apply_augmentation(torch.unsqueeze(x, dim=0)).squeeze(dim=0)
        return x, y

if __name__ == "__main__":
    d_path = sys.argv[1]
    c_path = sys.argv[2]

    s = SharedState()
    data = TimeSeriesTransformerDataset(d_path, c_path, s)

    samples, target = data[0]
    print(samples.shape)
    print(target.shape)