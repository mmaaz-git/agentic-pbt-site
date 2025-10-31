#!/usr/bin/env python3
"""Property-based test for ARGSORT_DEFAULTS duplicate key bug"""

from hypothesis import given, strategies as st
import pytest


def test_argsort_defaults_no_duplicate_keys():
    """Test that ARGSORT_DEFAULTS['kind'] has the expected value of 'quicksort'.

    According to NumPy documentation, the default value for the 'kind' parameter
    in numpy.argsort is 'quicksort'. The pandas compatibility layer should
    reflect this default value.

    However, due to a duplicate assignment in the source code, the value is
    incorrectly set to None.
    """
    from pandas.compat.numpy.function import ARGSORT_DEFAULTS

    # This assertion will fail because of the duplicate assignment bug
    assert ARGSORT_DEFAULTS["kind"] == "quicksort", \
        f"Expected 'quicksort', got {ARGSORT_DEFAULTS['kind']}"


if __name__ == "__main__":
    # Run the test
    try:
        test_argsort_defaults_no_duplicate_keys()
        print("✓ Test passed: ARGSORT_DEFAULTS['kind'] == 'quicksort'")
    except AssertionError as e:
        print("✗ Test failed:")
        print(f"  {e}")
        print()
        print("Bug details:")
        from pandas.compat.numpy.function import ARGSORT_DEFAULTS
        print(f"  Current ARGSORT_DEFAULTS: {ARGSORT_DEFAULTS}")
        print("  The 'kind' key is set twice in the source code:")
        print("    Line 138: ARGSORT_DEFAULTS['kind'] = 'quicksort'")
        print("    Line 140: ARGSORT_DEFAULTS['kind'] = None  (overwrites previous)")
        import sys
        sys.exit(1)