#!/usr/bin/env python3
"""Test to reproduce the PeriodArray.isin bug"""

import pandas as pd
from hypothesis import given, strategies as st
import traceback

# First, test the Hypothesis property-based test
@st.composite
def period_arrays(draw, min_size=1, max_size=50):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    freq = draw(st.sampled_from(['D', 'M', 'Y']))

    years = draw(st.lists(st.integers(min_value=2000, max_value=2030), min_size=size, max_size=size))
    months = draw(st.lists(st.integers(min_value=1, max_value=12), min_size=size, max_size=size))

    periods = [pd.Period(f'{y}-{m:02d}', freq=freq) for y, m in zip(years, months)]
    return pd.array(periods, dtype=f'period[{freq}]')

@given(period_arrays(min_size=1))
def test_period_array_isin_consistency(arr):
    first_val = None
    for val in arr:
        if pd.notna(val):
            first_val = val
            break

    if first_val is not None:
        result = arr.isin([first_val])

        for i, val in enumerate(arr):
            if pd.notna(val) and val == first_val:
                assert result[i]

print("Testing Hypothesis test...")
try:
    test_period_array_isin_consistency()
    print("Hypothesis test passed (unexpected)")
except Exception as e:
    print(f"Hypothesis test failed with error: {e}")
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Now test the minimal reproduction case
print("Testing minimal reproduction case...")
try:
    period = pd.Period('2000-01', freq='D')
    arr = pd.array([period], dtype='period[D]')
    print(f"Created PeriodArray: {arr}")
    print(f"Type of arr: {type(arr)}")

    print(f"\nAttempting arr.isin([{period}])...")
    result = arr.isin([period])
    print(f"Result: {result}")
    print("Minimal example passed (unexpected)")
except AttributeError as e:
    print(f"Got expected AttributeError: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Got unexpected error: {e}")
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Test IntegerArray comparison for consistency
print("Testing IntegerArray comparison...")
try:
    int_arr = pd.array([1, 2, 3], dtype='Int64')
    print(f"Created IntegerArray: {int_arr}")
    print(f"Type of int_arr: {type(int_arr)}")

    print(f"\nAttempting int_arr.isin([1])...")
    result = int_arr.isin([1])
    print(f"Result: {result}")
    print(f"Result type: {type(result)}")
    print("IntegerArray.isin worked correctly")
except Exception as e:
    print(f"IntegerArray.isin failed: {e}")
    traceback.print_exc()