from abc import ABC, abstractmethod
from typing import Optional

from torch.nn import Module

from utils import load_config
from src.shared_state import SharedState

class BaseModel(ABC, Module):
    def __init__(self, config_path: str, state: Optional[SharedState] = None, device: str = 'cpu', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._configs = load_config(config_path) if config_path != "" else {}
        self._state = state
        self._device = device

    def get_shared_state(self):
        return self._state
