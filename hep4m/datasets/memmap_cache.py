import numpy as np
from typing import Dict, Tuple

_GLOBAL_MMAPS: Dict[Tuple[str, str, int, int], np.memmap] = {}

def get_memmap(path: str, dtype: str, rows: int, cols: int) -> np.memmap:
    key = (path, dtype, rows, cols)
    mm = _GLOBAL_MMAPS.get(key)
    if mm is None:
        mm = np.memmap(path, dtype=dtype, mode='r', shape=(rows, cols))
        _GLOBAL_MMAPS[key] = mm
    return mm
