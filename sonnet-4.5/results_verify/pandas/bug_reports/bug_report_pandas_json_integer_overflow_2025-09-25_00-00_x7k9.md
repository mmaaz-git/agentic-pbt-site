# Bug Report: pandas.io.json Integer Overflow on Round-Trip

**Target**: `pandas.io.json.read_json` and `pandas.io.json.to_json`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When a DataFrame contains integer values outside the int64 range, `to_json()` successfully serializes them to JSON, but `read_json()` crashes with `ValueError: Value is too small` when attempting to deserialize, violating the documented round-trip property.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from io import StringIO

@given(
    data=st.lists(
        st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.integers(),
            min_size=1,
            max_size=5,
        ),
        min_size=1,
        max_size=20,
    ),
    orient=st.sampled_from(['records', 'index', 'columns', 'values', 'split']),
)
@settings(max_examples=100)
def test_read_json_to_json_roundtrip_dataframe(data, orient):
    df = pd.DataFrame(data)
    json_str = df.to_json(orient=orient)
    df_back = pd.read_json(StringIO(json_str), orient=orient)
```

**Failing input**: `data=[{'0': -9223372036854775809}], orient='split'` (or any orient)

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

df = pd.DataFrame([{'col': -9223372036854775809}])
json_str = df.to_json(orient='split')
df_back = pd.read_json(StringIO(json_str), orient='split')
```

Output:
```
ValueError: Value is too small
```

## Why This Is A Bug

The `read_json`/`to_json` round-trip property is explicitly documented in the pandas API. From the docstring:

> "Compatible JSON strings can be produced by `to_json()` with a corresponding orient value."

If `to_json()` can serialize a DataFrame, then `read_json()` should be able to deserialize it. However, DataFrames containing integers outside the int64 range (-9223372036854775808 to 9223372036854775807) break this invariant.

This violates user expectations: users can create valid DataFrames with arbitrary Python integers, serialize them to JSON successfully, but then fail to load them back. This can cause data loss in production systems.

## Fix

The issue arises because `ujson_loads` (the underlying JSON parser) cannot handle integers outside the int64 range, even though Python and pandas support arbitrary-precision integers.

Possible fixes:

1. **Validation on serialization**: Have `to_json()` validate that all integer values fit within int64 range and raise an error if they don't, preventing silent data corruption.

2. **String fallback**: When encountering integers outside int64 range, serialize them as strings with a special marker, then deserialize them back to Python integers.

3. **Error message improvement**: At minimum, provide a clearer error message that explains the int64 limitation and suggests workarounds.

The cleanest fix is option 1 - failing fast on serialization with a clear error message:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -260,6 +260,15 @@ class Writer(ABC):

     def write(self) -> str:
         iso_dates = self.date_format == "iso"
+        # Validate integer values are within int64 range
+        if isinstance(self.obj, (DataFrame, Series)):
+            for col in (self.obj if isinstance(self.obj, Series) else self.obj.values.flatten()):
+                if isinstance(col, int) and not (-9223372036854775808 <= col <= 9223372036854775807):
+                    raise ValueError(
+                        f"Integer value {col} is outside the int64 range "
+                        "and cannot be reliably round-tripped through JSON. "
+                        "Consider converting to float or string first."
+                    )
         return ujson_dumps(
             self.obj_to_write,
             orient=self.orient,
```