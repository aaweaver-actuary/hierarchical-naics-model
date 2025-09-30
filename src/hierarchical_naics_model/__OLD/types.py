from typing import Sequence, List, Mapping, Dict
import numpy as np

Strings = Sequence[str] | List[str]
Integers = Sequence[int] | List[int]
Mappings = Sequence[Mapping] | List[Mapping] | Dict[str, Mapping]
Arrays = Sequence[np.ndarray] | List[np.ndarray]
