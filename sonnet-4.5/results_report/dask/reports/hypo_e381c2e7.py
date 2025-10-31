#!/usr/bin/env python3
"""
Property-based test that discovers the bug in _normalize_and_strip_protocol
where root paths like "/" get converted to empty strings.
"""

from hypothesis import given, strategies as st, settings, example
from dask.dataframe.dask_expr.io.parquet import _normalize_and_strip_protocol

@given(st.text(min_size=1))
@settings(max_examples=500)
@example("/")
@example("///")
@example("s3:///")
def test_normalize_and_strip_protocol_no_empty_strings(path):
    result = _normalize_and_strip_protocol(path)

    assert len(result) == 1
    assert result[0] != "", f"Result should not be empty string for input {path!r}"

if __name__ == "__main__":
    # Run the test
    test_normalize_and_strip_protocol_no_empty_strings()