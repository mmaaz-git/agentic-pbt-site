#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings
from pandas.compat._optional import get_version
import types

@given(st.text())
@settings(max_examples=100)
def test_get_version_psycopg2_splits_on_whitespace(version_str):
    mock_module = types.ModuleType("psycopg2")
    mock_module.__version__ = version_str
    result = get_version(mock_module)
    assert isinstance(result, str)
    assert result == version_str.split()[0]

# Run the test
print("Running hypothesis test...")
try:
    test_get_version_psycopg2_splits_on_whitespace()
    print("Test passed!")
except Exception as e:
    print(f"Test failed: {e}")

# Test the specific failing input mentioned in the bug report
print("\nTesting specific failing input: version_str='\\r'")
mock_module = types.ModuleType("psycopg2")
mock_module.__version__ = "\r"
try:
    result = get_version(mock_module)
    print(f"Result: {repr(result)}")
except IndexError as e:
    print(f"IndexError raised: {e}")
    import traceback
    traceback.print_exc()