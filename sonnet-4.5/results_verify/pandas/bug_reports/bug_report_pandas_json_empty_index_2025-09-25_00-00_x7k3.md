# Bug Report: pandas.io.json Empty Index Type Change

**Target**: `pandas.io.json.read_json` / `pandas.io.json.to_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When round-tripping an empty DataFrame or Series with `orient='split'`, the index type changes from integer to float, violating the documented round-trip guarantee.

## Property-Based Test

```python
import pandas as pd
from io import StringIO
from hypothesis import given, settings, strategies as st
from pandas.testing import assert_frame_equal

@st.composite
def dataframes(draw):
    num_rows = draw(st.integers(min_value=0, max_value=20))
    num_cols = draw(st.integers(min_value=1, max_value=10))
    columns = [f"col_{i}" for i in range(num_cols)]
    data = {col: draw(st.lists(st.integers(), min_size=num_rows, max_size=num_rows))
            for col in columns}
    return pd.DataFrame(data)

@given(dataframes())
@settings(max_examples=200)
def test_dataframe_roundtrip_split(df):
    json_str = df.to_json(orient='split')
    df_roundtrip = pd.read_json(StringIO(json_str), orient='split')
    assert_frame_equal(df, df_roundtrip, check_dtype=False)
```

**Failing input**: `pd.DataFrame({"col_0": []})`

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

df = pd.DataFrame({"col_0": []})

print(f"Original index type: {df.index.inferred_type}")

json_str = df.to_json(orient='split')
df_roundtrip = pd.read_json(StringIO(json_str), orient='split')

print(f"Roundtrip index type: {df_roundtrip.index.inferred_type}")
print(f"Bug: {df.index.inferred_type} != {df_roundtrip.index.inferred_type}")
```

Output:
```
Original index type: integer
Roundtrip index type: floating
Bug: integer != floating
```

## Why This Is A Bug

1. **Violates documented behavior**: The `read_json` docstring states: "Compatible JSON strings can be produced by `to_json()` with a corresponding orient value." This implies round-trip compatibility.

2. **Realistic use case**: Empty DataFrames occur frequently in practice (e.g., filtering data with no matches, initializing schemas, reading empty sources).

3. **Type stability**: When the JSON contains `"index":[]`, the type cannot be inferred and pandas defaults to float64, but it should preserve the original integer type from the RangeIndex.

## Fix

The issue is in the `_json.py` file where empty arrays are parsed. When parsing split-oriented JSON with an empty index array, the code should check if the original DataFrame had a RangeIndex and preserve its dtype.

A potential fix would be to detect when the index array is empty and default to int64 for RangeIndex-compatible empty indices, or to include dtype metadata in the JSON schema for split orientation.