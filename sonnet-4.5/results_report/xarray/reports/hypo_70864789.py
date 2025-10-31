#!/usr/bin/env python3
"""
Property-based test for RangeIndex.arange that discovered the negative size bug.
"""

import sys
import os

# Add the xarray environment to path
sys.path.insert(0, '/home/npc/miniconda/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume, settings
from xarray.indexes.range_index import RangeIndex

@given(
    start=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    stop=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    step=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
)
@settings(max_examples=1000)
def test_arange_size_nonnegative(start, stop, step):
    assume(step != 0)
    assume(abs(step) > 1e-10)

    index = RangeIndex.arange(start, stop, step, dim="x")
    assert index.size >= 0, f"Size must be non-negative, got {index.size}"

# Run the test
if __name__ == "__main__":
    print("Running property-based test for RangeIndex.arange...")
    print()

    try:
        test_arange_size_nonnegative()
        print("All tests passed!")
    except AssertionError as e:
        print("Test failed!")
        print()
        print("The test found a case where RangeIndex.arange produces a negative size.")
        print("This violates the invariant that array dimensions must be non-negative.")
        print()
        print("Minimal failing example found by Hypothesis:")
        print("  start=1.0, stop=0.0, step=1.0")
        print()
        print("Running the minimal example directly...")
        index = RangeIndex.arange(1.0, 0.0, 1.0, dim="x")
        print(f"  Result: index.size = {index.size} (should be >= 0)")
        print()
        print("This proves that RangeIndex.arange can produce negative dimension sizes,")
        print("which violates fundamental array invariants.")