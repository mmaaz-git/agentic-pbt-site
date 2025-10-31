# Bug Report: pandas.io.json Empty DataFrame Index Dtype Changes During Round-Trip

**Target**: `pandas.io.json.read_json` and `pd.DataFrame.to_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When an empty DataFrame with integer index is serialized to JSON and read back, the index dtype changes from int64 to float64, violating the round-trip property.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from io import StringIO
import pandas as pd

@settings(max_examples=500)
@given(
    st.data(),
    st.sampled_from(['split', 'records', 'index', 'columns', 'values', 'table'])
)
def test_dataframe_json_round_trip(data, orient):
    """Round-trip: read_json(df.to_json(orient=x), orient=x) should preserve data"""

    ncols = data.draw(st.integers(min_value=1, max_value=5))
    nrows = data.draw(st.integers(min_value=0, max_value=10))

    columns = [f'col_{i}' for i in range(ncols)]
    df_data = {col: data.draw(st.lists(st.integers(), min_size=nrows, max_size=nrows)) for col in columns}
    df = pd.DataFrame(df_data)

    if orient in ['index', 'columns']:
        assume(df.index.is_unique)
    if orient in ['index', 'columns', 'records']:
        assume(df.columns.is_unique)

    json_str = df.to_json(orient=orient)
    recovered = pd.read_json(StringIO(json_str), orient=orient)

    pd.testing.assert_frame_equal(recovered, df)
```

**Failing input**: `pd.DataFrame({'col_0': []})`

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

df = pd.DataFrame({'a': []})
print(f"Original index dtype: {df.index.dtype}")
print(f"Original index: {df.index.tolist()}")

json_str = df.to_json(orient='split')
print(f"JSON: {json_str}")

recovered = pd.read_json(StringIO(json_str), orient='split')
print(f"Recovered index dtype: {recovered.index.dtype}")
print(f"Recovered index: {recovered.index.tolist()}")
print(f"Dtypes match: {df.index.dtype == recovered.index.dtype}")
```

Output:
```
Original index dtype: int64
Original index: []
JSON: {"columns":["a"],"index":[],"data":[]}
Recovered index dtype: float64
Recovered index: []
Dtypes match: False
```

## Why This Is A Bug

Empty DataFrames are common in data processing workflows:
- Result of filtering operations that match no rows
- Initialization before appending data
- Edge cases in data pipelines

The index dtype matters for:
- Concatenation and merging with other DataFrames
- Type consistency in data pipelines
- Proper functioning of downstream operations

When JSON contains an empty array `[]`, pandas cannot infer the dtype and defaults to float64, but it should preserve the original int64 dtype from the serialization metadata. The 'split' orient already includes structure information, so it should be possible to encode dtype information.

## Fix

The issue is in how `read_json` handles empty arrays during dtype inference. When the index array is empty and no explicit dtype is provided, pandas defaults to float64. The fix should:

1. For 'split' orient: Include dtype metadata in the JSON output
2. For 'table' orient: This already works correctly as it includes schema information
3. For other orients: Use int64 as default for empty index (matching DataFrame's default behavior)

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -xxx,x +xxx,x @@ def _parse_index(index_data):
     if len(index_data) == 0:
-        # Current: Returns float64 index
-        return pd.Index(index_data)
+        # Fixed: Returns int64 index (matching DataFrame default)
+        return pd.Index(index_data, dtype='int64')
     return pd.Index(index_data)
```