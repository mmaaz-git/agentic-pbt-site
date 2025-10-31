#!/usr/bin/env python3
"""Test script to reproduce the pandas qcut bug"""

import pandas as pd
from hypothesis import given, strategies as st, assume, settings

# First, test the minimal reproduction case
print("=" * 60)
print("Testing minimal reproduction case")
print("=" * 60)

data = [0.0, 0.0, 1.0, 2.0, 3.0]
print(f"Input data: {data}")
result = pd.qcut(data, q=4, duplicates='drop')
print(f"Result categories: {result.categories.tolist()}")
print(f"Result values: {result.tolist()}")
value_counts = result.value_counts()
print(f"Quartile counts: {value_counts.tolist()}")
print(f"Quartile distribution: {dict(value_counts)}")

# Check the assertion from the bug report
if len(value_counts) > 1:
    min_count = value_counts.min()
    max_count = value_counts.max()
    difference = max_count - min_count
    print(f"Max count: {max_count}, Min count: {min_count}, Difference: {difference}")
    if difference > 1:
        print(f"ERROR: Quartile sizes too uneven (difference={difference})")
    else:
        print(f"OK: Quartile sizes are balanced (difference={difference})")

print("\n" + "=" * 60)
print("Testing with Hypothesis property-based test")
print("=" * 60)

# Now test with the hypothesis test
@given(st.lists(st.floats(0, 100, allow_nan=False), min_size=4, max_size=50))
@settings(max_examples=100)
def test_qcut_quartile_property(data):
    assume(len(set(data)) >= 4)

    result = pd.qcut(data, q=4, duplicates='drop')
    value_counts = result.value_counts()

    if len(value_counts) > 1:
        min_count = value_counts.min()
        max_count = value_counts.max()
        assert max_count - min_count <= 1, f"Quartile sizes too uneven: {value_counts.tolist()}, data={data}"

try:
    test_qcut_quartile_property()
    print("Hypothesis test passed!")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")
except Exception as e:
    print(f"Unexpected error in hypothesis test: {e}")