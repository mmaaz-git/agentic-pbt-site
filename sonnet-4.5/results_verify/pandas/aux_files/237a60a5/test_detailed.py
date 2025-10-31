#!/usr/bin/env python3
"""More detailed test to see specific failing cases"""

import pandas as pd
from hypothesis import given, strategies as st, assume, settings, example

# Test with specific failing cases
@given(st.lists(st.floats(0, 100, allow_nan=False, allow_infinity=False), min_size=4, max_size=50))
@settings(max_examples=10)
@example([0.0, 0.0, 1.0, 2.0, 3.0])  # Known failing case
def test_qcut_quartile_property(data):
    assume(len(set(data)) >= 4)

    print(f"\nTesting with data: {data[:10]}..." if len(data) > 10 else f"\nTesting with data: {data}")
    print(f"Data length: {len(data)}, Unique values: {len(set(data))}")

    result = pd.qcut(data, q=4, duplicates='drop')
    value_counts = result.value_counts()

    print(f"Number of bins created: {len(value_counts)}")
    print(f"Bin counts: {value_counts.tolist()}")

    if len(value_counts) > 1:
        min_count = value_counts.min()
        max_count = value_counts.max()
        difference = max_count - min_count
        print(f"Max-Min difference: {difference}")

        try:
            assert difference <= 1, f"Quartile sizes too uneven"
            print("✓ PASS")
        except AssertionError:
            print("✗ FAIL: Uneven distribution")
            raise

test_qcut_quartile_property()