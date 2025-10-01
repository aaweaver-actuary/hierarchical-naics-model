from __future__ import annotations

from typing import TypedDict


class QuotesRow(TypedDict):
    """Schema for a single quotes row."""

    submission_id: str
    is_written: int
    naics: str
    zip: str
    quote_date: str  # ISO8601
