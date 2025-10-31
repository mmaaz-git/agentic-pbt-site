# Bug Report: pandas JSON Integer Overflow Crash

**Target**: `pandas.api.typing.JsonReader` / `pandas.read_json`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When a Series or DataFrame containing integers outside the int64 range (< -2^63 or >= 2^63) is serialized to JSON and then deserialized, ujson raises a `ValueError: Value is too small` or `Value is too large` error.

## Property-Based Test

```python
import pandas as pd
from io import StringIO
from hypothesis import given, strategies as st, settings


@given(st.integers())
def test_series_json_roundtrip(value):
    s = pd.Series([value])
    json_str = s.to_json(orient='split')
    result = pd.read_json(StringIO(json_str), typ='series', orient='split', convert_dates=False)
    pd.testing.assert_series_equal(result, s)
```

**Failing input**: `value=-9_223_372_036_854_775_809` (or any value outside int64 range)

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

value = -9_223_372_036_854_775_809
s = pd.Series([value])
print(f"Original Series: {s.tolist()}, dtype={s.dtype}")

json_str = s.to_json(orient='split')
print(f"JSON: {json_str}")

result = pd.read_json(StringIO(json_str), typ='series', orient='split')
```

Output:
```
Original Series: [-9223372036854775809], dtype=object
JSON: {"name":null,"index":[0],"data":[-9223372036854775809]}
ValueError: Value is too small
```

## Why This Is A Bug

While integers outside int64 range are stored with object dtype, pandas allows them to be serialized to JSON but crashes when attempting to deserialize the same JSON. This asymmetry violates the principle that serialization and deserialization should be inverse operations. Users working with large integer IDs or counts (common in databases) would encounter this crash.

## Fix

Add validation in `to_json` to either:
1. Warn/error when serializing integers outside int64 range, or
2. Add a fallback in `read_json` to handle these values gracefully (e.g., read as object dtype or use Python's arbitrary-precision integers)

Option 2 is preferable for backward compatibility:

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -1360,7 +1360,11 @@ class SeriesParser(Parser):

     def _parse(self) -> None:
-        data = ujson_loads(self.json, precise_float=self.precise_float)
+        try:
+            data = ujson_loads(self.json, precise_float=self.precise_float)
+        except ValueError as e:
+            if "too small" in str(e) or "too large" in str(e):
+                import json
+                data = json.loads(self.json)
+            else:
+                raise

         if self.orient == "split":
```