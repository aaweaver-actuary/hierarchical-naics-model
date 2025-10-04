from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..types import Integers


class ConversionModelStrategy(ABC):
    """Abstract base class for hierarchical conversion model strategies."""

    @abstractmethod
    def build_model(
        self,
        *,
        y: np.ndarray,
        naics_levels: np.ndarray,
        zip_levels: np.ndarray,
        naics_group_counts: Integers,
        zip_group_counts: Integers,
    ) -> Any:
        """Construct a probabilistic model for the provided hierarchical data."""

    @abstractmethod
    def sample_posterior(
        self,
        model: Any,
        *,
        draws: int,
        tune: int,
        chains: int,
        cores: int,
        target_accept: float | None = None,
        progressbar: bool = False,
        random_seed: int | None = None,
    ) -> Any:
        """Run inference for the supplied model and return posterior samples."""
