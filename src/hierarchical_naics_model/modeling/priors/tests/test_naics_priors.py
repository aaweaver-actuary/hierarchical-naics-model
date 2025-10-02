from hierarchical_naics_model.modeling.priors.naics import NaicsPriors
from hierarchical_naics_model.modeling.priors.utils import NormalParams


def test_naics_priors_instantiation():
    priors = NaicsPriors()
    assert isinstance(priors, NaicsPriors)
    assert priors.use_student_t_level0 is False
    assert priors.level0 is None

    # Test with custom values
    normal = NormalParams(mu=0.0, sigma=1.0)
    priors2 = NaicsPriors(use_student_t_level0=True, level0=normal)
    assert priors2.use_student_t_level0 is True
    assert priors2.level0 == normal
