"""Series of mathematical functions used in the modeling code. Platform-agnostic wrappers for PyMC and NumPy."""

from typing import Optional, Union
from types import ModuleType

import numpy as np

try:
    import pymc as pm
    import pytensor.tensor as pt

    _has_pymc = True
except ImportError:
    pm: Optional[ModuleType] = None
    pt: Optional[ModuleType] = None
    _has_pymc = False


def exp(
    x: Union[np.ndarray, float, int],
) -> Union[np.ndarray, float]:
    """Platform-agnostic exp wrapper."""
    if (
        _has_pymc
        and pt is not None
        and hasattr(pt, "TensorVariable")
        and isinstance(x, pt.TensorVariable)
    ):
        return pt.exp(x)  # type: ignore[attr-defined]
    return np.exp(x)


def sigmoid(
    x: Union[np.ndarray, float, int],
) -> Union[np.ndarray, float]:
    """Platform-agnostic sigmoid wrapper."""
    if (
        _has_pymc
        and pt is not None
        and hasattr(pt, "TensorVariable")
        and isinstance(x, pt.TensorVariable)
    ):
        return pt.sigmoid(x)  # type: ignore[attr-defined]
    return 1 / (1 + np.exp(-x))
