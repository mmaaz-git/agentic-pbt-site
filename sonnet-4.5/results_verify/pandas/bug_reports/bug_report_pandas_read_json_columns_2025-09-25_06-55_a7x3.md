# Bug Report: pandas.read_json Column Name Type Corruption

**Target**: `pandas.read_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`pd.read_json` incorrectly converts numeric string column names to integers when all column names are numeric, breaking JSON roundtrip serialization.

## Property-Based Test

```python
from hypothesis import assume, given, settings, strategies as st
import pandas as pd
from io import StringIO


@given(st.lists(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), min_size=1, max_size=5), min_size=1, max_size=20))
@settings(max_examples=500)
def test_json_roundtrip(records):
    df = pd.DataFrame(records)
    assume(len(df.columns) > 0)

    json_str = df.to_json(orient='records')
    df_read = pd.read_json(StringIO(json_str), orient='records')

    pd.testing.assert_frame_equal(df, df_read)
```

**Failing input**: `records=[{'0': 0}]`

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

df = pd.DataFrame({'0': [1, 2], '1': [3, 4]})
print('Original columns:', df.columns.tolist())
print('Column types:', [type(c).__name__ for c in df.columns])

json_str = df.to_json(orient='records')
df_back = pd.read_json(StringIO(json_str), orient='records')

print('After roundtrip:', df_back.columns.tolist())
print('Column types:', [type(c).__name__ for c in df_back.columns])

assert df.columns.tolist() == ['0', '1']
assert df_back.columns.tolist() == [0, 1]
```

Output:
```
Original columns: ['0', '1']
Column types: ['str', 'str']
After roundtrip: [0, 1]
Column types: ['int', 'int']
```

## Why This Is A Bug

1. **Breaks roundtrip serialization**: `df.to_json().read_json() != df` violates the fundamental expectation that serialization should be reversible
2. **JSON spec violation**: JSON object keys are always strings, so they should remain strings when read
3. **Inconsistent behavior**: If there's a mix of numeric and non-numeric column names, they stay as strings. But if all are numeric, they get converted to integers
4. **Silent corruption**: No warning or error is raised, leading to subtle bugs in downstream code that expects string column names

Example of inconsistency:
```python
pd.read_json('[{"0":1}]', orient='records').columns.tolist()
pd.read_json('[{"0":1,"name":"a"}]', orient='records').columns.tolist()
```

## Fix

The bug is in `read_json`'s column name inference logic. When orient='records', the function should either:

1. **Always preserve column names as strings** (recommended, matches JSON spec)
2. **Add a parameter to control type inference** for column names
3. **Document the current behavior** if it's intentional (though this seems unlikely)

Recommended approach - modify the column name parsing to preserve string types:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -somewhere in FrameParser
-    columns = _infer_types(column_names)
+    columns = pd.Index(column_names, dtype=object)
```

The exact location would be in the `FrameParser` class's column name handling for orient='records'.