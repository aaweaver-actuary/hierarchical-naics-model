from .noncentered_normal_rv import _noncentered_normal_rv
from .disable_random_effects import _disable_random_effects


def _get_naics_base(naics_group_counts, naics_mu_0, naics_sigma_0):
    return (
        _disable_random_effects("naics", naics_group_counts)
        if int(naics_group_counts[0]) == 1
        else _noncentered_normal_rv(
            "naics_base",
            mu=naics_mu_0,
            sigma=naics_sigma_0,
            shape=(naics_group_counts[0],),
            dims=("NAICS_L0",),
        )
    )
