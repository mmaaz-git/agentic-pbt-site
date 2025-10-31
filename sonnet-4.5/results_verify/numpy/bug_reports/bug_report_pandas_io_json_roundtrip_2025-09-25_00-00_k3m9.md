# Bug Report: pandas.io.json Round-Trip String Index Conversion

**Target**: `pandas.io.json.read_json` and `pandas.io.json.to_json`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a DataFrame has string indices or columns that look like numbers (e.g., '0', '1'), round-tripping through `to_json()` and `read_json()` with `orient='index'` or `orient='columns'` silently converts them to integers, violating the documented claim that "Compatible JSON strings can be produced by `to_json()` with a corresponding orient value."

## Property-Based Test

```python
import pandas as pd
import io
from hypothesis import given, strategies as st, settings
from pandas.testing import assert_frame_equal


@given(
    st.data(),
    st.sampled_from(['index', 'columns'])
)
@settings(max_examples=200)
def test_json_roundtrip_preserves_string_axes(data, orient):
    num_rows = data.draw(st.integers(min_value=1, max_value=5))
    num_cols = data.draw(st.integers(min_value=1, max_value=5))

    df = pd.DataFrame({
        f'col_{i}': [j for j in range(num_rows)]
        for i in range(num_cols)
    })

    if orient == 'index':
        df.index = df.index.astype(str)
    else:
        df.columns = df.columns.astype(str)

    json_str = df.to_json(orient=orient)
    df_roundtrip = pd.read_json(io.StringIO(json_str), orient=orient)

    assert_frame_equal(df, df_roundtrip, check_dtype=False)
```

**Failing input**: `orient='index'`, `num_rows=1`, `num_cols=1`

## Reproducing the Bug

```python
import pandas as pd
import io

df = pd.DataFrame({'col_0': [0]})
df.index = df.index.astype(str)

print("Original index:", list(df.index), "- dtype:", df.index.dtype)

json_str = df.to_json(orient='index')
df_roundtrip = pd.read_json(io.StringIO(json_str), orient='index')

print("Roundtrip index:", list(df_roundtrip.index), "- dtype:", df_roundtrip.index.dtype)
print("Equal?", df.equals(df_roundtrip))
```

Output:
```
Original index: ['0'] - dtype: object
Roundtrip index: [0] - dtype: int64
Equal? False
```

## Why This Is A Bug

1. **Documentation claims compatibility**: The `read_json` docstring states: "Compatible JSON strings can be produced by `to_json()` with a corresponding orient value." This implies that round-tripping should preserve the DataFrame structure.

2. **Inconsistent behavior**: The conversion only happens when ALL labels look numeric. If even one label is non-numeric (e.g., mixed index `['0', 'a', '2']`), the entire index stays as strings. This inconsistency is confusing.

3. **Silent data corruption**: String indices like `['0', '1', '2']` are silently converted to integers `[0, 1, 2]`, changing the data type without warning (except in the mixed case).

4. **Affects both index and columns**: The bug affects both `orient='index'` (converting row indices) and `orient='columns'` (converting column names).

## Fix

The issue stems from `read_json`'s default `convert_axes=True` parameter, which attempts to infer better dtypes for axes. While this may be useful in some cases, it breaks the round-trip property.

**Workaround**: Users can preserve the round-trip by passing `convert_axes=False`:

```python
df_roundtrip = pd.read_json(io.StringIO(json_str), orient='index', convert_axes=False)
```

**Proposed fix**: Change the default behavior when the orient suggests a round-trip scenario, or update documentation to clearly warn that round-tripping may not preserve string axes that look like numbers.

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -read_json function

     # Option 1: Document the limitation clearly
-    Compatible JSON strings can be produced by ``to_json()`` with a
-    corresponding orient value.
+    Compatible JSON strings can be produced by ``to_json()`` with a
+    corresponding orient value. Note: by default, ``convert_axes=True``
+    may convert string axes that appear numeric to numeric types. To
+    preserve exact round-trip compatibility, use ``convert_axes=False``.

     # Option 2: Change default behavior for specific orients
     # When orient is 'index' or 'columns', default convert_axes to False