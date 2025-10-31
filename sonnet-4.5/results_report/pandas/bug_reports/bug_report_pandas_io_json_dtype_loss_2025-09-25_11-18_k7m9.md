# Bug Report: pandas.io.json Float Dtype Lost During Round-Trip

**Target**: `pandas.io.json.read_json` and `pd.DataFrame.to_json` / `pd.Series.to_json`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a Series or DataFrame with float64 dtype containing whole number values is serialized to JSON and read back, pandas incorrectly infers the dtype as int64, causing data corruption and breaking the fundamental round-trip property.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from io import StringIO
import pandas as pd

@settings(max_examples=200)
@given(
    st.data(),
    st.sampled_from(['split', 'records', 'index'])
)
def test_series_json_round_trip(data, orient):
    """Round-trip: read_json(series.to_json(orient=x), orient=x, typ='series') should preserve data"""

    nrows = data.draw(st.integers(min_value=1, max_value=10))
    values = data.draw(st.lists(
        st.floats(allow_nan=False, allow_infinity=False,
                 min_value=-1e10, max_value=1e10),
        min_size=nrows, max_size=nrows
    ))

    series = pd.Series(values)

    if orient == 'index':
        assume(series.index.is_unique)

    json_str = series.to_json(orient=orient)
    recovered = pd.read_json(StringIO(json_str), orient=orient, typ='series')

    pd.testing.assert_series_equal(recovered, series, check_index_type=False)
```

**Failing input**: `pd.Series([1.0, 2.0, 3.0])`

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

series = pd.Series([1.0, 2.0, 3.0])
print(f"Original dtype: {series.dtype}")

json_str = series.to_json(orient='split')
print(f"JSON: {json_str}")

recovered = pd.read_json(StringIO(json_str), orient='split', typ='series')
print(f"Recovered dtype: {recovered.dtype}")
print(f"Dtypes match: {series.dtype == recovered.dtype}")
```

Output:
```
Original dtype: float64
JSON: {"name":null,"index":[0,1,2],"data":[1.0,2.0,3.0]}
Recovered dtype: int64
Dtypes match: False
```

## Why This Is A Bug

This violates the fundamental round-trip property that `read_json(data.to_json(orient=x), orient=x)` should preserve the original data structure and types. The pandas documentation for `read_json` states that it can "Convert a JSON string to pandas object" and the `to_json` docstring shows examples where round-tripping is expected to work.

Float values with no fractional part (like 1.0, 2.0) are common in real-world data:
- Financial data (prices that happen to be whole dollars)
- Scientific measurements (rounded values)
- Aggregated statistics

When the dtype changes from float64 to int64, it can cause:
- Type errors in downstream operations expecting floats
- Different behavior in arithmetic (integer division vs float division)
- Failure to detect the silent data corruption

## Fix

The issue is in `read_json`'s dtype inference logic. When `dtype=True` (the default), pandas should preserve the JSON numeric type rather than inferring based on values. Since JSON distinguishes `1.0` from `1`, pandas should respect this distinction.

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -xxx,x +xxx,x @@ def _infer_dtype_from_value(value):
-    # Current: Infer int64 if all floats are whole numbers
-    if isinstance(value, float) and value.is_integer():
-        return 'int64'
+    # Fixed: Preserve float type even if value is a whole number
+    # JSON distinguishes 1.0 from 1, so should pandas
     return 'float64'
```

Alternatively, the JSON serialization could include explicit type metadata when using orient='table' to preserve dtypes, but this should work for all orient modes.