from hierarchical_naics_model.types import Integers


def test_integers_type_alias():
    ints: Integers = [1, 2, 3]
    assert isinstance(ints, list)
    ints2: Integers = (4, 5, 6)
    assert isinstance(ints2, tuple) or isinstance(ints2, list)
