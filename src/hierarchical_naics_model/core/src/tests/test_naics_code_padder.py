from hierarchical_naics_model.core.src.naics_code_padder import NaicsCodePadder
import pytest


@pytest.fixture(scope="session")
def padder():
    return NaicsCodePadder()


BEFORE = [
    "11",
    "111",
    "1111",
    "11111",
    "111110",
    "1111100",
    "21",
    "2111",
    "2111100",
    "23",
]

AFTER = [
    "110000",
    "111000",
    "111100",
    "111110",
    "111110",
    "err",
    "210000",
    "211100",
    "err",
    "230000",
]


@pytest.mark.parametrize("before, after", zip(BEFORE, AFTER))
def test_naics_code_padder(padder, before, after):
    if after == "err":
        with pytest.raises(ValueError) as _:
            padder(before)
    else:
        expected = after
        actual = padder(before)
        assert actual == expected, f"for {before}, expected {expected} but got {actual}"
