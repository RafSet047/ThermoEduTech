import sys
from typing import *

import torch
from src.dataset import ThermoDataset
from src.shared_state import SharedState

import warnings
warnings.filterwarnings("ignore")

from loguru import logger

class ThermoTransformerDataset(ThermoDataset):
    """
    A dataset class specifically designed for transformer-based models handling thermographic data.
    It extends the ThermoDataset class and includes functionalities for handling numerical and
    categorical features as well as target variables.

    Attributes:
        __numerical_features (List[str]): List of numerical feature column names.
        __categorical_features (List[str]): List of categorical feature configurations.
        __target_feature (List[str]): List containing the name of the target feature.
        __X_num (torch.Tensor): Tensor containing numerical features.
        __X_cat (torch.Tensor): Tensor containing categorical features.
        __Y (torch.Tensor): Tensor containing target values.

    Methods:
        __parse_configs(): Parses the configuration to set up numerical and categorical features.
        __prepare_features(): Prepares tensors for numerical features, categorical features, and target values.
        get_dataset() -> Any: Returns the dataset comprising numerical and categorical features and targets.
        __getitem__(index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: Returns a sample from the dataset.
    """
    def __init__(self, data_path: str, configs_path: str, shared_state: Optional[SharedState] = None, device: str = 'cpu') -> None:
        """
        Initializes the ThermoTransformerDataset with paths for data and configs, shared state, and device.

        Args:
            data_path (str): Path to the CSV file containing the dataset.
            configs_path (str): Path to the configuration file.
            shared_state (Optional[SharedState]): An optional object for sharing state between different components.
            device (str): The device type ('cpu' or 'gpu') for data processing.
        """
        super().__init__(data_path, configs_path, shared_state, device)

        self.__parse_configs()
        self.__prepare_features()

        self._shared_state.num_categories = self._num_categories
        self._shared_state.num_continious = self._num_continious

    def __parse_configs(self) -> None:
        """
        Parses the configuration file to extract numerical and categorical features,
        as well as the target feature.
        """
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
        """
        Prepares tensors for numerical features, categorical features, and target values
        from the dataframe.
        """
        self._df[self.__target_feature] = self._df[self.__target_feature].shift(-1) # ground truth is the next hour
        self._df.dropna(axis=0, inplace=True) # removing the last row
        self._N = self._df.shape[0]

        self.__X_num = torch.as_tensor(self._df[self.__numerical_features].values, dtype=torch.float32, device=self._device)
        self.__X_cat = torch.as_tensor(self._df[self.__cat_features].values, dtype=torch.int32, device=self._device)
        self.__Y = torch.as_tensor(self._df[self.__target_feature].values, dtype=torch.float32, device=self._device)

    def prepare_data(self):
        return super().prepare_data()

    def get_dataset(self) -> Any:
        """
        Returns the dataset comprising numerical and categorical features and targets.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the tensors for numerical
            and categorical features, and the target tensor.
        """
        return (self.__X_num, self.__X_cat), self.__Y

    def __getitem__(self, index) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns a sample from the dataset at the specified index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: A tuple containing numerical and categorical features, and the target.
        """
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

