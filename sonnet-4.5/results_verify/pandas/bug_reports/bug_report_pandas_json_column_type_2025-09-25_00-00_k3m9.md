# Bug Report: pandas.io.json Column Type Conversion

**Target**: `pandas.io.json.read_json` and `pandas.io.json.to_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a DataFrame has string column names that look like numbers (e.g., "0", "1"), the round-trip `read_json(to_json(df))` converts these string column names to integers, violating the expected round-trip property.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from io import StringIO
from pandas.io.json import read_json, to_json

@given(
    st.lists(
        st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
            min_size=1,
            max_size=5
        ),
        min_size=1,
        max_size=10
    )
)
def test_read_json_to_json_roundtrip_records(data):
    df = pd.DataFrame(data)
    json_str = to_json(None, df, orient='records')
    result = read_json(StringIO(json_str), orient='records')
    pd.testing.assert_frame_equal(result, df, check_dtype=False)
```

**Failing input**: `data=[{'0': 0}]`

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO
from pandas.io.json import read_json, to_json

df = pd.DataFrame([{'0': 0}])
print("Original columns:", df.columns.tolist())
print("Original column type:", type(df.columns[0]))

json_str = to_json(None, df, orient='records')
result = read_json(StringIO(json_str), orient='records')

print("Result columns:", result.columns.tolist())
print("Result column type:", type(result.columns[0]))
print("Are they equal?", df.columns[0] == result.columns[0])
```

Output:
```
Original columns: ['0']
Original column type: <class 'str'>
Result columns: [0]
Result column type: <class 'numpy.int64'>
Are they equal? False
```

## Why This Is A Bug

The pandas documentation for `read_json` shows examples of round-trip operations that should preserve the DataFrame structure. When column names are strings, they should remain strings after serialization and deserialization. This bug breaks the fundamental expectation that `read_json(to_json(df)) == df`.

The issue occurs because `read_json` has `convert_axes=True` by default, which attempts to convert string axes to their "proper" dtypes. However, this conversion is too aggressive - it converts string "0" to integer 0, which changes the semantics of the DataFrame.

## Fix

The fix is to make axis conversion more conservative. When `convert_axes=True`, the code should preserve string column names unless there's a strong reason to convert them. One approach:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -1296,8 +1296,11 @@ class Parser:
                 pass

         # if we have an index, we want to preserve dtypes
-        if name == "index" and len(data):
-            if self.orient == "split":
+        if is_axis and len(data):
+            # For axes (index/columns), only convert for split orient
+            # to preserve column names like "0" as strings
+            if self.orient == "split":
+                return data, False
+            elif name == "columns":
                 return data, False

         return data, converted
```

However, a simpler workaround for users is to use `convert_axes=False`:
```python
result = read_json(StringIO(json_str), orient='records', convert_axes=False)
```