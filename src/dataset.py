from abc import ABC, abstractmethod
import pandas as pd
from typing import *

from torch.utils.data import Dataset
from utils import load_config
from src.shared_state import SharedState

class ThermoDataset(ABC, Dataset):
    def __init__(self, data_path: str, configs_path: str, shared_state: Optional[SharedState] = None, device: str = 'cpu') -> None:
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
        return self._df.copy()

    @property
    def configs(self) -> Optional[Dict]:
        return self._configs.copy()

    def get_shared_state(self):
        return self._shared_state

    def __len__(self) -> int:
        return self._N

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        pass