#!/usr/bin/env python3
"""Property-based test using hypothesis - fixed version"""

from hypothesis import given, strategies as st, settings
from pandas.arrays import SparseArray
import numpy as np

def test_specific_case(data):
    """Test a specific data case"""
    sparse = SparseArray(data)
    dense = sparse.to_dense()

    try:
        sparse_argmin = sparse.argmin()
        numpy_argmin = np.argmin(dense)
        assert sparse_argmin == numpy_argmin, f"argmin mismatch: sparse={sparse_argmin}, numpy={numpy_argmin}"
        print(f"  argmin: sparse={sparse_argmin}, numpy={numpy_argmin} ✓")
    except Exception as e:
        print(f"  argmin failed: {type(e).__name__}: {e}")
        raise

    try:
        sparse_argmax = sparse.argmax()
        numpy_argmax = np.argmax(dense)
        assert sparse_argmax == numpy_argmax, f"argmax mismatch: sparse={sparse_argmax}, numpy={numpy_argmax}"
        print(f"  argmax: sparse={sparse_argmax}, numpy={numpy_argmax} ✓")
    except Exception as e:
        print(f"  argmax failed: {type(e).__name__}: {e}")
        raise

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50))
@settings(max_examples=100)
def test_argmin_argmax_consistent_with_dense(data):
    sparse = SparseArray(data)
    dense = sparse.to_dense()

    sparse_argmin = sparse.argmin()
    numpy_argmin = np.argmin(dense)
    assert sparse_argmin == numpy_argmin

    sparse_argmax = sparse.argmax()
    numpy_argmax = np.argmax(dense)
    assert sparse_argmax == numpy_argmax

if __name__ == "__main__":
    print("Running property-based test...")

    # First test the specific failing case mentioned in the report
    print("\nTesting specific failing case: [0]")
    try:
        test_specific_case([0])
        print("✓ Test passed for [0]")
    except Exception as e:
        print(f"✗ Test failed for [0]")

    print("\nTesting specific failing case: [0, 0, 0]")
    try:
        test_specific_case([0, 0, 0])
        print("✓ Test passed for [0, 0, 0]")
    except Exception as e:
        print(f"✗ Test failed for [0, 0, 0]")

    print("\nTesting case with non-fill values: [0, 1, 2]")
    try:
        test_specific_case([0, 1, 2])
        print("✓ Test passed for [0, 1, 2]")
    except Exception as e:
        print(f"✗ Test failed for [0, 1, 2]: {e}")

    print("\nRunning hypothesis tests...")
    try:
        test_argmin_argmax_consistent_with_dense()
        print("✓ All hypothesis tests passed")
    except Exception as e:
        print(f"✗ Hypothesis test failed: {e}")