from typing import *
from dataclasses import dataclass

@dataclass
class SharedState:
    num_categories: Optional[Tuple[int]] = None
    num_continious: Optional[int] = None