#!/usr/bin/env python3
"""Test the purported bug in pandas categorical null value handling in interchange protocol."""

import pandas as pd
import numpy as np
from pandas.api.interchange import from_dataframe

# Test case from the bug report
df = pd.DataFrame({"col": pd.Categorical(["cat1", None])})
print("Original DataFrame:")
print(df)
print(f"Original values: {df['col'].tolist()}")
print(f"Original dtype: {df['col'].dtype}")

# Perform round-trip through interchange protocol
interchange_obj = df.__dataframe__()
result = from_dataframe(interchange_obj)
print("\nAfter round-trip:")
print(result)
print(f"Result values: {result['col'].tolist()}")
print(f"Result dtype: {result['col'].dtype}")

# Check if null value is preserved
original_has_null = pd.isna(df['col'].iloc[1])
result_has_null = pd.isna(result['col'].iloc[1])
print(f"\nOriginal has null at position 1: {original_has_null}")
print(f"Result has null at position 1: {result_has_null}")

# Also run the property-based test
from hypothesis import given, strategies as st, settings
import pandas.testing as tm

@given(
    st.lists(st.sampled_from(["cat1", "cat2", "cat3", None]), min_size=0, max_size=100),
    st.booleans()
)
@settings(max_examples=10)
def test_round_trip_categorical(cat_list, ordered):
    df = pd.DataFrame({"col": pd.Categorical(cat_list, ordered=ordered)})
    result = from_dataframe(df.__dataframe__())
    try:
        tm.assert_frame_equal(result, df)
        print(f"✓ PASSED: cat_list={cat_list[:3]}... ordered={ordered}")
    except AssertionError as e:
        print(f"✗ FAILED: cat_list={cat_list} ordered={ordered}")
        print(f"  Error: {str(e)[:100]}...")
        return False
    return True

print("\n\nRunning hypothesis property-based test:")
test_round_trip_categorical()