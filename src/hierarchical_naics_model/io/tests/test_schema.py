from __future__ import annotations

from hierarchical_naics_model.io.schema import QuotesRow


def test_quotes_row_typed_dict_allows_expected_fields():
    row: QuotesRow = {
        "submission_id": "ABC123",
        "is_written": 1,
        "naics": "52",
        "zip": "12345",
        "quote_date": "2024-01-01",
    }
    assert row["is_written"] == 1
