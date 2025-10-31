#!/usr/bin/env python3
"""Test the DateAttribute bug"""

import sys
import numpy as np
from scipy.io.arff._arffread import DateAttribute
from hypothesis import given, strategies as st, assume

# Test the specific bug - line 276 always evaluates to True
print("Testing the bug report...")
print()

# First, prove that "yy" is always truthy
print(f"bool('yy') = {bool('yy')}  # This is always True!")
print()

# Test the reproduction code
test_pattern = "date MM-dd"
result_fmt, result_unit = DateAttribute._get_date_format(test_pattern)

print(f"Pattern: {test_pattern}")
print(f"Result format: {result_fmt}")
print(f"Result unit: {result_unit}")
print()

print("Expected: datetime_unit should never be 'Y' during processing")
print("Actual: Line 276 'elif \"yy\":' always evaluates to True")
print("        causing datetime_unit='Y' to be set incorrectly")
print("        before being overwritten by later 'dd' check")
print()

# Test various patterns
test_patterns = [
    "date MM-dd",  # No year
    "date dd-MM",  # No year
    "date HH:mm:ss",  # No year, time only
    "date yyyy-MM-dd",  # Has yyyy
    "date yy-MM-dd",  # Has yy
]

print("Testing various patterns:")
for pattern in test_patterns:
    try:
        fmt, unit = DateAttribute._get_date_format(pattern)
        print(f"  {pattern:<20} -> unit={unit}, fmt={fmt}")
    except ValueError as e:
        print(f"  {pattern:<20} -> ValueError: {e}")

print()

# Run the hypothesis test
@given(st.text(alphabet='MdHms-/: ', min_size=1, max_size=50))
def test_date_format_without_year_shouldnt_set_year_unit(pattern_body):
    """
    Test that date patterns without year components don't incorrectly
    set datetime_unit to 'Y' during intermediate processing.

    This property would fail if we could observe intermediate state,
    but passes in practice because later components overwrite the bug.
    """
    assume('y' not in pattern_body.lower())
    assume(any(x in pattern_body for x in ['M', 'd', 'H', 'm', 's']))

    pattern_str = f"date {pattern_body}"

    try:
        date_fmt, datetime_unit = DateAttribute._get_date_format(pattern_str)
        assert datetime_unit != "Y", \
            f"Pattern {pattern_body} has no year but datetime_unit is 'Y'"
    except ValueError:
        pass

print("Running hypothesis test (100 examples)...")
try:
    test_date_format_without_year_shouldnt_set_year_unit()
    print("Hypothesis test passed (bug is masked by overwrites)")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")