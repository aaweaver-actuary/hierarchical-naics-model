from dataclasses import dataclass

from .utils import Priors


@dataclass
class NaicsPriors:
    use_student_t_level0: bool = False

    level0: Priors = None
