#!/usr/bin/env python3
"""Property-based test for format_bytes using Hypothesis"""

from hypothesis import given, strategies as st, settings
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=100)
def test_format_bytes_length_constraint(n):
    """Property: For all values < 2**60, output is always <= 10 characters (documented)"""
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)} > 10"

# Run the test
if __name__ == "__main__":
    try:
        test_format_bytes_length_constraint()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        # Let's find the minimum value that violates the constraint
        print("\nSearching for the minimum value that violates the constraint...")
        for exp in range(50, 61):
            val = 2**exp
            result = format_bytes(val)
            if len(result) > 10:
                print(f"First violation at 2**{exp}: format_bytes({val}) = '{result}' (length: {len(result)})")
                break

        # Check around 1000 * 2**50
        print("\nChecking around 1000 * 2**50:")
        for multiplier in [999, 999.9, 999.99, 1000]:
            val = int(multiplier * 2**50)
            result = format_bytes(val)
            print(f"format_bytes({multiplier} * 2**50) = '{result}' (length: {len(result)})")