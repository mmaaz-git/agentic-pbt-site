#!/usr/bin/env python3
"""Hypothesis test for scipy.constants.precision() non-negativity property."""

from hypothesis import given, strategies as st, settings
from scipy.constants import find, precision

def test_precision_is_non_negative():
    """Test that precision() returns non-negative values for all physical constants."""
    all_keys = find(None, disp=False)
    failures = []

    for key in all_keys:
        prec = precision(key)
        if prec < 0:
            failures.append((key, prec))

    if failures:
        print(f"Found {len(failures)} constants with negative precision:")
        for key, prec in failures[:5]:  # Show first 5 failures
            print(f"  precision('{key}') = {prec}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more")

        # Raise assertion with the first failing example
        key, prec = failures[0]
        assert prec >= 0, f"precision('{key}') = {prec}, should be non-negative"
    else:
        print("All precision values are non-negative (test passed)")

if __name__ == "__main__":
    test_precision_is_non_negative()