#!/usr/bin/env python3
"""Hypothesis test for _excel2num whitespace bug"""

import string
from hypothesis import given, strategies as st, example
from pandas.io.excel._util import _excel2num
import pytest

@given(st.text(min_size=0, max_size=5))
@example(' ')  # single space
@example('')   # empty string
@example('\t') # tab
@example('\n') # newline
def test_excel2num_invalid_chars_raise_error(text):
    """_excel2num should raise ValueError for invalid characters"""
    # If the text contains any non-alphabetic character or is empty after stripping
    if not text.strip() or not all(c.isalpha() for c in text.strip()):
        with pytest.raises(ValueError, match="Invalid column name"):
            _excel2num(text)
    else:
        # Valid Excel column names should not raise
        result = _excel2num(text)
        assert isinstance(result, int)
        assert result >= 0

if __name__ == "__main__":
    # Run the test
    test_excel2num_invalid_chars_raise_error()