# Bug Report: pandas.io.json Empty DataFrame Index Type Change

**Target**: `pandas.io.json.read_json` / `pandas.io.json.to_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Empty DataFrames lose their RangeIndex type during JSON round-trip with `orient='split'`, becoming a float64 Index instead.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from pandas.io.json import read_json, to_json
import io

@given(
    data=st.lists(
        st.fixed_dictionaries({
            'a': st.integers(min_value=-1000, max_value=1000),
            'b': st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        }),
        min_size=0,
        max_size=20
    )
)
@settings(max_examples=100)
def test_roundtrip_orient_split(data):
    df = pd.DataFrame(data)
    json_str = to_json(None, df, orient='split')
    result = read_json(io.StringIO(json_str), orient='split')
    pd.testing.assert_frame_equal(result, df)
```

**Failing input**: `[]` (empty list)

## Reproducing the Bug

```python
import pandas as pd
from pandas.io.json import read_json, to_json
import io

df = pd.DataFrame([])
print(f"Original index: {df.index}")
print(f"Original index type: {type(df.index).__name__}")
print(f"Original index dtype: {df.index.dtype}")

json_str = to_json(None, df, orient='split')
print(f"\nJSON: {json_str}")

result = read_json(io.StringIO(json_str), orient='split')
print(f"\nResult index: {result.index}")
print(f"Result index type: {type(result.index).__name__}")
print(f"Result index dtype: {result.index.dtype}")
```

Output:
```
Original index: RangeIndex(start=0, stop=0, step=1)
Original index type: RangeIndex
Original index dtype: int64

JSON: {"columns":[],"index":[],"data":[]}

Result index: Index([], dtype='float64')
Result index type: Index
Result index dtype: float64
```

## Why This Is A Bug

1. The index type changes from `RangeIndex` to `Index`
2. The dtype changes from `int64` to `float64`
3. The `inferred_type` changes from "integer" to "floating"

This violates the documented round-trip property. While the DataFrames are functionally equivalent for empty data, this breaks strict equality checks and can cause issues in code that depends on index types.

## Fix

The issue occurs because `to_json` with `orient='split'` serializes an empty RangeIndex as `[]`, losing the type information. When `read_json` deserializes this, it creates a generic Index with default float64 dtype.

**Proposed solution**:
1. When serializing with `orient='split'`, preserve RangeIndex metadata if the index is a RangeIndex
2. OR default to int64 for empty index arrays instead of float64
3. OR recommend using `orient='table'` for exact round-trips (which already preserves this correctly)