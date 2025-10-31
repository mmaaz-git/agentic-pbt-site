# Bug Report: pandas.io.json Integer Overflow with Table Orient

**Target**: `pandas.io.json.read_json` with `orient='table'`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When round-tripping a DataFrame with uint64 values exceeding int64 max through `to_json(orient='table')` and `read_json(orient='table')`, the values silently overflow to negative int64 values, causing silent data corruption.

## Property-Based Test

```python
import pandas as pd
import io
from hypothesis import given, strategies as st, settings
from pandas.testing import assert_frame_equal


@given(st.data())
@settings(max_examples=200)
def test_table_orient_roundtrip(data):
    num_rows = data.draw(st.integers(min_value=1, max_value=5))
    num_cols = data.draw(st.integers(min_value=1, max_value=5))

    columns = [f'col_{i}' for i in range(num_cols)]

    df_data = {}
    for col in columns:
        df_data[col] = data.draw(
            st.lists(
                st.integers(),
                min_size=num_rows,
                max_size=num_rows
            )
        )

    df = pd.DataFrame(df_data)

    json_str = df.to_json(orient='table')
    df_roundtrip = pd.read_json(io.StringIO(json_str), orient='table')

    assert_frame_equal(df, df_roundtrip, check_dtype=False)
```

**Failing input**: DataFrame with value `9223372036854775808` (2^63)

## Reproducing the Bug

```python
import pandas as pd
import io

df = pd.DataFrame({'col': [9223372036854775808]})

print(f"Original value: {df['col'][0]}")
print(f"Original dtype: {df['col'].dtype}")

json_str = df.to_json(orient='table')
df_roundtrip = pd.read_json(io.StringIO(json_str), orient='table')

print(f"Roundtrip value: {df_roundtrip['col'][0]}")
print(f"Roundtrip dtype: {df_roundtrip['col'].dtype}")
print(f"Data corrupted: {df['col'][0] != df_roundtrip['col'][0]}")
```

Output:
```
Original value: 9223372036854775808
Original dtype: uint64
Roundtrip value: -9223372036854775808
Roundtrip dtype: int64
Data corrupted: True
```

## Why This Is A Bug

1. **Silent data corruption**: A value of 9,223,372,036,854,775,808 becomes -9,223,372,036,854,775,808 with no warning or error.

2. **Violates round-trip property**: The documentation claims compatibility between `to_json` and `read_json` with matching orient values.

3. **Schema doesn't preserve signedness**: The table schema uses generic "integer" type without distinguishing uint64 from int64.

4. **Other orients work correctly**: The 'split' and 'records' orients handle uint64 values correctly.

## Fix

The table schema should distinguish between signed and unsigned integer types, or read_json should respect the actual JSON values instead of forcing them into int64.

```diff
--- a/pandas/io/json/_table_schema.py
+++ b/pandas/io/json/_table_schema.py

 def set_default_names(data):
     # When building the table schema, distinguish uint64 from int64
-    elif dtype.kind in ('i', 'u'):
-        return 'integer'
+    elif dtype.kind == 'i':
+        return 'integer'
+    elif dtype.kind == 'u':
+        return 'unsignedInteger'  # or use a range constraint