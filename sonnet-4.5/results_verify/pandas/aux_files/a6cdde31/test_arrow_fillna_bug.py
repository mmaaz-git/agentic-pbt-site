#!/usr/bin/env python3
"""Test script to reproduce the ArrowExtensionArray fillna bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

# First, run the hypothesis test
print("=" * 60)
print("Running Hypothesis test...")
print("=" * 60)

from hypothesis import given, strategies as st, settings
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

@given(st.data())
@settings(max_examples=500)
def test_arrow_extension_array_fillna_length(data):
    values = data.draw(st.lists(st.one_of(st.integers(min_value=-1000, max_value=1000), st.none()), min_size=1, max_size=100))
    arr = ArrowExtensionArray(pa.array(values))
    fill_value = data.draw(st.integers(min_value=-1000, max_value=1000))

    result = arr.fillna(fill_value)

    assert len(result) == len(arr)

# Run the hypothesis test
try:
    test_arrow_extension_array_fillna_length()
    print("Hypothesis test passed unexpectedly!")
except Exception as e:
    print(f"Hypothesis test failed as expected with: {e}")

# Now test the specific failing case
print("\n" + "=" * 60)
print("Running specific failing case: values=[None], fill_value=0")
print("=" * 60)

try:
    arr = ArrowExtensionArray(pa.array([None]))
    print(f"Created array: {arr}")
    print(f"Array dtype: {arr.dtype}")
    print(f"PyArrow type: {arr._pa_array.type}")

    print("\nAttempting to call fillna(0)...")
    result = arr.fillna(0)
    print(f"Unexpected success! Result: {result}")
except Exception as e:
    print(f"\nCaught exception: {type(e).__name__}: {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()