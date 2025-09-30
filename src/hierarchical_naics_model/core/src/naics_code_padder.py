from typing import List


class NaicsCodePadder:
    """Utility class for padding NAICS codes to a standard 6-digit format.

    If a code is shorter than 6 digits, it will be right-padded with the '0' character.
    If a code is longer than 6 digits, it will raise an error.
    """

    def __init__(self, target_length: int = 6, pad_char: str = "0"):
        self.target_length = target_length
        self.pad_char = pad_char

    def pad(self, code: str) -> str:
        if len(code) > self.target_length:
            raise ValueError(
                f"NAICS code '{code}' exceeds maximum length of {self.target_length}."
            )
        return code.ljust(self.target_length, self.pad_char)

    def __call__(self, codes: List[str]) -> List[str]:
        return [self.pad(code) for code in codes]
