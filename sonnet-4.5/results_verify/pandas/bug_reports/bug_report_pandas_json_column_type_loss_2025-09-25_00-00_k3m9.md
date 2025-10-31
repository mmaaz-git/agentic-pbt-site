# Bug Report: pandas.io.json Column/Index Type Loss on Round-Trip

**Target**: `pandas.io.json.read_json` and `pandas.io.json.to_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

String column names and index values that look like numbers (e.g., '0', '123') are silently converted to integers during JSON round-trip with certain orient values, violating the documented round-trip property and causing data corruption.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from io import StringIO
from pandas.testing import assert_frame_equal

@given(
    data=st.lists(
        st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.one_of(st.integers(), st.text()),
            min_size=1,
            max_size=5,
        ),
        min_size=1,
        max_size=20,
    ),
    orient=st.sampled_from(['records', 'columns']),
)
def test_read_json_to_json_roundtrip(data, orient):
    df = pd.DataFrame(data)
    json_str = df.to_json(orient=orient)
    df_back = pd.read_json(StringIO(json_str), orient=orient)
    assert_frame_equal(df, df_back)
```

**Failing input**: `data=[{'0': 0}], orient='records'`

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

df = pd.DataFrame([{'0': 123}])
print(f"Original columns: {df.columns.tolist()}")
print(f"Column type: {type(df.columns[0])}")

json_str = df.to_json(orient='records')
print(f"JSON: {json_str}")

df_back = pd.read_json(StringIO(json_str), orient='records')
print(f"Round-trip columns: {df_back.columns.tolist()}")
print(f"Column type: {type(df_back.columns[0])}")
print(f"Columns equal? {df.columns.equals(df_back.columns)}")
```

Output:
```
Original columns: ['0']
Column type: <class 'str'>
JSON: [{"0":123}]
Round-trip columns: [0]
Column type: <class 'numpy.int64'>
Columns equal? False
```

## Why This Is A Bug

The `convert_axes` parameter in `read_json()` defaults to `True` for most orient values, which attempts to convert axes to "proper dtypes". However, this conversion is too aggressive and converts numeric-looking strings to integers, even though the original DataFrame had string column names.

This violates the round-trip property documented in the API and can cause serious issues:
- Column lookups by name will fail: `df['0']` works, but `df_back['0']` raises KeyError
- Data integrity is silently corrupted without warning
- Merges and joins on these DataFrames will fail

The issue affects both column names and index values with orient='records' and orient='index'.

## Fix

The conversion logic in `_convert_axes()` (lines 1189-1207 in `_json.py`) needs to be more conservative. It should preserve the original types from the JSON rather than trying to infer "better" types.

For the specific case of string keys in JSON objects, they should remain as strings:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -1296,8 +1296,12 @@ class Parser:
             return data, False

         # if we have an index, we want to preserve dtypes
-        if name == "index" and len(data):
-            if self.orient == "split":
+        if name == "index":
+            # For orient='split', preserve exact index from JSON
+            # For other orients, index comes from JSON object keys which are always strings
+            if self.orient == "split":
+                return data, False
+            elif self.orient in ("index", "columns"):
                 return data, False

         return data, converted
```

Alternatively, the default for `convert_axes` could be changed to `False` to match user expectations that round-tripping preserves exact data types.