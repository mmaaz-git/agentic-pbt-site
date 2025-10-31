# Bug Report: pandas.io.json - Column Name Data Loss in JSON Round-Trip

**Target**: `pandas.io.json` (specifically `pd.read_json` and `df.to_json`)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a DataFrame with all-numeric string column names is serialized to JSON and deserialized back, the column names are incorrectly converted from strings to integers, causing data loss. Critically, column names like '00' are converted to 0, losing the leading zero.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from io import StringIO


@given(
    columns=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5, unique=True),
    num_rows=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=200)
def test_json_roundtrip_column_names(columns, num_rows):
    data = {col: list(range(num_rows)) for col in columns}
    df = pd.DataFrame(data)

    json_str = df.to_json(orient='split')
    result = pd.read_json(StringIO(json_str), orient='split')

    assert list(result.columns) == list(df.columns)
```

**Failing input**: `columns=['0']` (or any list where all elements are numeric strings)

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

df = pd.DataFrame({'00': [1, 2], '0': [3, 4]})
print("Original columns:", list(df.columns))

json_str = df.to_json(orient='split')
result = pd.read_json(StringIO(json_str), orient='split')

print("Result columns:", list(result.columns))
print("Data loss: '00' became", result.columns[0])
```

**Output:**
```
Original columns: ['00', '0']
Result columns: [0, 0]
Data loss: '00' became 0
```

**Additional examples:**
- `['0']` → `[0]`
- `['0', '1']` → `[0, 1]`
- `['00']` → `[0]` (DATA LOSS!)
- `['0', 'a']` → `['0', 'a']` (preserved when mixed with non-numeric)

## Why This Is A Bug

1. **Silent data corruption**: Column name '00' becomes 0, losing the leading zero
2. **Type mismatch**: User explicitly created string columns, they should remain strings
3. **Inconsistent behavior**: Mixed columns (e.g., ['0', 'a']) are preserved correctly, but all-numeric columns are converted
4. **JSON format integrity**: The JSON output correctly stores `"columns":["00","0"]` as strings, so the bug is in `read_json`'s deserialization
5. **Round-trip property violation**: DataFrame serialization should preserve column names exactly
6. **No clear workaround**: The `convert_axes` parameter doesn't prevent this conversion

## Fix

The fix should prevent automatic conversion of column names that are numeric strings to integers. One approach:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -xxx,x +xxx,x @@ def _parse_with_pandas(...):
     if orient == "split":
         decoded = loads(json, ...)
         if "columns" in decoded:
-            # Don't auto-convert column names to numeric types
-            decoded["columns"] = _maybe_convert_axes(decoded["columns"], convert_axes)
+            # Preserve column names as-is to avoid data loss
+            # Column names should remain the type the user specified
+            pass
```

Alternatively, modify `_maybe_convert_axes` to skip conversion when it would cause data loss (e.g., '00' → 0).