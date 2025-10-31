# Bug Report: pandas.io.json Parse Error for Large Integers

**Target**: `pandas.io.json.read_json` (ALL orients)
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When a DataFrame contains very large negative integers (below int64 min: -9,223,372,036,854,775,809), `to_json()` successfully creates JSON with the value for ALL orients, but `read_json()` crashes with "ValueError: Value is too small" when trying to parse it back, regardless of which orient is used.

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

**Failing input**: DataFrame with value `-9223372036854775809` (int64 min - 1), any orient

## Reproducing the Bug

```python
import pandas as pd
import io

df = pd.DataFrame({'col': [-9223372036854775809]})

for orient in ['split', 'records', 'index', 'columns', 'values', 'table']:
    try:
        json_str = df.to_json(orient=orient)
        df_roundtrip = pd.read_json(io.StringIO(json_str), orient=orient)
        print(f"{orient}: Success")
    except ValueError as e:
        print(f"{orient}: Error - {e}")
```

Output:
```
split: Error - Value is too small
records: Error - Value is too small
index: Error - Value is too small
columns: Error - Value is too small
values: Error - Value is too small
table: Error - Value is too small
```

## Why This Is A Bug

1. **Asymmetric behavior**: `to_json` succeeds for all orients but `read_json` fails for all orients, breaking the round-trip property.

2. **Underlying ujson limitation**: The error comes from the ujson library's inability to parse integers outside the int64 range (-2^63 to 2^63-1), but pandas doesn't handle this gracefully.

3. **Poor error message**: "Value is too small" doesn't clearly indicate what the issue is or how to fix it.

4. **Valid JSON rejected**: The JSON is valid and contains the correct data, but pandas refuses to read it.

## Fix

Pandas should catch ValueError from ujson_loads when it encounters integers outside int64 range and either:
1. Fall back to a standard JSON parser that can handle arbitrary precision integers
2. Provide a clearer error message explaining the limitation
3. Automatically convert to object dtype or Python int type

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py

 def _parse(self):
-    data = ujson_loads(json, precise_float=self.precise_float)
+    try:
+        data = ujson_loads(json, precise_float=self.precise_float)
+    except ValueError as e:
+        if "too small" in str(e) or "too big" in str(e):
+            # Fall back to standard json parser for large integers
+            import json
+            data = json.loads(json)
+        else:
+            raise