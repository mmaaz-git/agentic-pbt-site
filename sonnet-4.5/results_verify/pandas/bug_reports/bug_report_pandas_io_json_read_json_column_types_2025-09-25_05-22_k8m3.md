# Bug Report: pandas.io.json.read_json converts numeric string column names to integers

**Target**: `pandas.io.json.read_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`read_json` converts string column names that consist only of digits (e.g., `'0'`, `'00'`, `'123'`) to integer types during JSON round-trip, even when the JSON explicitly contains string values. This violates the round-trip property and causes data loss (e.g., leading zeros).

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
import io

@given(col_name=st.from_regex(r'^[0-9]+$', fullmatch=True))
def test_numeric_string_column_names_preserved(col_name):
    df = pd.DataFrame({col_name: [1, 2, 3]})

    json_str = df.to_json(orient='split')
    df_roundtrip = pd.read_json(io.StringIO(json_str), orient='split')

    read_col = df_roundtrip.columns[0]

    assert col_name == str(read_col), \
        f"Column name not preserved: {col_name!r} != {read_col!r}"
```

**Failing input**: `'00'`

## Reproducing the Bug

```python
import pandas as pd
import io

df = pd.DataFrame({'00': [1, 2, 3]})
print(f"Original column: {df.columns[0]!r}")

json_str = df.to_json(orient='split')
print(f"JSON: {json_str}")

df_roundtrip = pd.read_json(io.StringIO(json_str), orient='split')
print(f"Roundtrip column: {df_roundtrip.columns[0]!r}")
```

Expected output:
```
Original column: '00'
JSON: {"columns":["00"],"index":[0,1,2],"data":[[1],[2],[3]]}
Roundtrip column: '00'
```

Actual output:
```
Original column: '00'
JSON: {"columns":["00"],"index":[0,1,2],"data":[[1],[2],[3]]}
Roundtrip column: 0
```

## Why This Is A Bug

1. **Data loss**: String column names like `'00'`, `'007'` lose their leading zeros when converted to integers, and there's no way to recover the original value.

2. **Round-trip property violation**: The JSON explicitly contains string values (`"columns":["00"]`), but `read_json` incorrectly interprets them as integers.

3. **Type inconsistency**: The original DataFrame has string column names, but after round-trip it has integer column names, breaking type consistency.

4. **Unexpected behavior**: Users expect that `read_json(df.to_json(...))` should reconstruct the original DataFrame, but this fundamental assumption is violated for numeric string column names.

## Fix

The issue is that `read_json` is attempting to convert JSON string values that look like numbers into numeric types. For column names (and index values), this conversion should not happen, or should be optional via a parameter.

A patch could add a parameter to control this behavior:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -800,6 +800,7 @@ def read_json(
     storage_options: StorageOptions | None = None,
     dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
     engine: JSONEngine = "ujson",
+    convert_axes: bool = False,
 ) -> DataFrame | Series | JsonReader:
```

Alternatively, the fix could be in the JSON parser to not convert column/index names that are explicitly strings in the JSON.