#!/usr/bin/env python3
"""Hypothesis test for scipy.io.arff DateAttribute bug."""

from hypothesis import given, strategies as st, example
from scipy.io.arff._arffread import DateAttribute

@given(st.sampled_from([
    "date ''",           # Empty pattern - should raise ValueError
    "date 'XXX'",        # Invalid pattern - should raise ValueError
    "date 'abc'",        # Invalid pattern - should raise ValueError
    "date '123'",        # Invalid pattern - should raise ValueError
]))
@example("date ''")
def test_invalid_date_formats_raise_error(date_string):
    """
    Test that invalid date formats with no valid date/time components raise ValueError.
    Due to the bug at line 276, these invalid patterns are incorrectly accepted.
    """
    pattern = date_string.split("'")[1] if "'" in date_string else ""

    # Check if pattern contains any valid date/time component
    valid_components = ['yyyy', 'yy', 'MM', 'dd', 'HH', 'mm', 'ss']
    has_valid_component = any(comp in pattern for comp in valid_components)

    if not has_valid_component:
        # Should raise ValueError for invalid patterns
        try:
            date_format, datetime_unit = DateAttribute._get_date_format(date_string)
            # If we get here without exception, the bug is present
            assert False, f"Expected ValueError for invalid pattern '{pattern}', but got Format: {date_format}, Unit: {datetime_unit}"
        except ValueError as e:
            # This is the expected behavior
            assert "Invalid or unsupported date format" in str(e)

if __name__ == "__main__":
    test_invalid_date_formats_raise_error()