from typing import Union
from dataclasses import dataclass


@dataclass
class NormalParams:
    mu: float
    sigma: float


@dataclass
class StudentTParams:
    nu: float
    mu: float
    sigma: float


@dataclass
class HalfNormalParams:
    sigma: float


Priors = Union[NormalParams, StudentTParams, HalfNormalParams]
