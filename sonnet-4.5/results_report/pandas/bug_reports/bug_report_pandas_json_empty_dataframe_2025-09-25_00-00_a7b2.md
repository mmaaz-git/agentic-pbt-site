# Bug Report: pandas.io.json Empty DataFrame Index Type Corruption

**Target**: `pandas.io.json.read_json` and `pandas.DataFrame.to_json`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When an empty DataFrame is serialized with `to_json()` and deserialized with `read_json()`, the index type changes from `RangeIndex` with integer inferred_type to `Index` with floating inferred_type, violating the round-trip property.

## Property-Based Test

```python
import io
import pandas as pd
from hypothesis import given, strategies as st, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes

@given(
    data_frames(
        columns=[
            column("int_col", dtype=int, elements=st.integers(min_value=-1e10, max_value=1e10)),
            column("float_col", dtype=float, elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10)),
            column("str_col", dtype=str, elements=st.text(min_size=0, max_size=100)),
        ],
        index=range_indexes(min_size=0, max_size=100),
    )
)
@settings(max_examples=100)
def test_json_round_trip_basic(df):
    json_str = df.to_json()
    df_round_trip = pd.read_json(io.StringIO(json_str))
    pd.testing.assert_frame_equal(df, df_round_trip)
```

**Failing input**: Empty DataFrame with columns `[int_col, float_col, str_col]` and empty RangeIndex

## Reproducing the Bug

```python
import io
import pandas as pd

df = pd.DataFrame(columns=["int_col", "float_col", "str_col"])

print(f"Original index: {df.index}")
print(f"Original inferred_type: {df.index.inferred_type}")

json_str = df.to_json()

df_round_trip = pd.read_json(io.StringIO(json_str))

print(f"Round-trip index: {df_round_trip.index}")
print(f"Round-trip inferred_type: {df_round_trip.index.inferred_type}")

assert df.index.inferred_type == df_round_trip.index.inferred_type
```

## Why This Is A Bug

The round-trip property is fundamental to serialization: deserializing serialized data should produce the original data structure. The index type is part of the DataFrame's structure and should be preserved. This bug causes:

1. `pd.testing.assert_frame_equal()` to fail when comparing the original and round-tripped DataFrames
2. Potential downstream issues when code relies on index type properties
3. Inconsistent behavior between empty and non-empty DataFrames

## Fix

The issue appears to be in how `read_json` infers the index type when the DataFrame is empty. When there are no rows, the index should be created as a RangeIndex rather than defaulting to a generic Index with float dtype.

A high-level fix would be:
1. In `read_json`, detect when the resulting DataFrame is empty
2. If empty and the index appears to be a sequential range, create a RangeIndex instead of a generic Index
3. Ensure the index dtype inference matches the non-empty case