#!/usr/bin/env python3
"""Test script to reproduce the attrs cmp_using error message bug"""

import attrs
from attrs import cmp_using
import pytest

# First test: the pytest test from the bug report
def test_cmp_using_error_message():
    with pytest.raises(ValueError, match="eq must be define"):
        cmp_using(lt=lambda a, b: a < b)

# Second test: manual reproduction
print("Reproducing the bug manually:")
try:
    cmp_using(lt=lambda a, b: a < b)
except ValueError as e:
    print(f"Error message: {e}")
    print(f"\nChecking for typos:")
    error_str = str(e)
    if "define is order" in error_str:
        print("✓ Found 'define is order' (should be 'defined in order')")
    if "eq must be define" in error_str:
        print("✓ Found 'eq must be define' (should be 'eq must be defined')")

# Run the pytest test
print("\nRunning pytest test:")
try:
    test_cmp_using_error_message()
    print("✓ Pytest test passed - error message contains 'eq must be define'")
except AssertionError:
    print("✗ Pytest test failed - error message doesn't match expected pattern")