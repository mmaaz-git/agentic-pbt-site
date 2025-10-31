#!/usr/bin/env python3
"""Test to reproduce the NBitBase deprecation warning bug"""

import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st

# First test: Property-based test using hypothesis
@given(st.just("NBitBase"))
def test_deprecated_attribute_warns(attr_name):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        getattr(npt, attr_name)
        assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
        assert issubclass(w[0].category, DeprecationWarning), f"Expected DeprecationWarning, got {w[0].category if w else 'no warnings'}"

# Second test: Direct reproduction
def test_direct_access():
    print("\nDirect access test:")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        obj = npt.NBitBase
        print(f"Warnings captured: {len(w)}")
        print(f"Expected: 1, Actual: {len(w)}")
        if len(w) > 0:
            print(f"Warning message: {w[0].message}")
            print(f"Warning category: {w[0].category}")
        else:
            print("No warnings were captured!")
        return len(w) == 1

if __name__ == "__main__":
    # Run the direct test
    result = test_direct_access()
    print(f"\nDirect test {'PASSED' if result else 'FAILED'}")

    # Run the hypothesis test
    print("\nRunning hypothesis test...")
    try:
        test_deprecated_attribute_warns()
        print("Hypothesis test FAILED - no assertion error raised")
    except AssertionError as e:
        print(f"Hypothesis test FAILED as expected - {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")