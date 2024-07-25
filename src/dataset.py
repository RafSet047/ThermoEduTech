from abc import ABC, abstractmethod
import pandas as pd
from typing import *

from torch.utils.data import Dataset
from utils import load_config
from src.shared_state import SharedState

class ThermoDataset(ABC, Dataset):
    """
    An abstract base class for handling datasets with thermographic data. This class extends
    PyTorch's Dataset and includes additional functionalities like shared state and device management.

    Attributes:
        _df (pd.DataFrame): The dataframe containing the dataset.
        _N (int): The number of samples in the dataset.
        _configs (Dict): Configuration parameters loaded from the provided configs path.
        _shared_state (Optional[SharedState]): An optional object for sharing state between different components.
        _device (str): The device type ('cpu' or 'gpu') for data processing.
        _num_categories (Optional[Tuple[int]]): The number of categories in the dataset (if applicable).
        _num_continious (Optional[int]): The number of continuous features in the dataset (if applicable).

    Methods:
        df() -> Optional[pd.DataFrame]: Returns a copy of the dataset dataframe.
        configs() -> Optional[Dict]: Returns a copy of the configuration parameters.
        get_shared_state(): Returns the shared state object.
        __len__() -> int: Returns the number of samples in the dataset.
        get_dataset() -> Any: Abstract method to be implemented by subclasses for retrieving the dataset.
        __getitem__(index: int) -> Any: Abstract method to be implemented by subclasses for accessing data samples.
    """
    def __init__(self, data_path: str, configs_path: str, shared_state: Optional[SharedState] = None, device: str = 'cpu') -> None:
        """
        Initializes the ThermoDataset with data path, configuration path, shared state, and device.

        Args:
            data_path (str): Path to the CSV file containing the dataset.
            configs_path (str): Path to the configuration file.
            shared_state (Optional[SharedState]): An optional object for sharing state between different components.
            device (str): The device type ('cpu' or 'gpu') for data processing.
        """
        super().__init__()
        self._df = pd.read_csv(data_path)
        self._N = self._df.shape[0]
        self._configs = load_config(configs_path)
        self._shared_state = shared_state
        self._device = device

        self._num_categories: Optional[Tuple[int]] = None
        self._num_continious: Optional[int] = None

    @property
    def df(self) -> Optional[pd.DataFrame]:
        """
        Returns a copy of the dataset dataframe.

        Returns:
            Optional[pd.DataFrame]: A copy of the dataframe containing the dataset.
        """
        return self._df.copy()

    @property
    def configs(self) -> Optional[Dict]:
        """
        Returns a copy of the configuration parameters.

        Returns:
            Optional[Dict]: A copy of the configuration parameters loaded from the config file.
        """
        return self._configs.copy()

    def get_shared_state(self):
        """
        Returns the shared state object.

        Returns:
            Optional[SharedState]: The shared state object.
        """
        return self._shared_state

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self._N

    @abstractmethod
    def get_dataset(self) -> Any:
        """
        Abstract method to be implemented by subclasses for retrieving the dataset.

        Returns:
            Any: The dataset, specific to the implementation.
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """
        Abstract method to be implemented by subclasses for accessing data samples.

        Args:
            index (int): The index of the data sample to retrieve.

        Returns:
            Any: The data sample corresponding to the given index.
        """
        pass