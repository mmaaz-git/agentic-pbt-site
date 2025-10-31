# Bug Report: pandas.io.json ujson_dumps Default Precision Too Low

**Target**: `pandas.io.json.ujson_dumps`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`ujson_dumps()` with default precision (10) truncates large floats causing them to overflow to infinity when deserialized with `ujson_loads()`, violating the round-trip property. Standard library `json.dumps()` handles these values correctly.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.json import ujson_dumps, ujson_loads
import math


@given(st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=500)
def test_ujson_roundtrip_preserves_finiteness(value):
    serialized = ujson_dumps(value)
    deserialized = ujson_loads(serialized)

    if math.isfinite(value):
        assert math.isfinite(deserialized), \
            f"Round-trip should preserve finiteness: {value} -> {serialized} -> {deserialized}"
```

**Failing input**: `1.7976931345e+308`

## Reproducing the Bug

```python
from pandas.io.json import ujson_dumps, ujson_loads
import json
import math

value = 1.7976931345e+308

stdlib_result = json.loads(json.dumps(value))
print(f"stdlib json:  {value} -> {stdlib_result} (finite: {math.isfinite(stdlib_result)})")

ujson_result = ujson_loads(ujson_dumps(value))
print(f"ujson:        {value} -> {ujson_result} (finite: {math.isfinite(ujson_result)})")

assert stdlib_result == value
assert ujson_result != value
```

Output:
```
stdlib json:  1.7976931345e+308 -> 1.7976931345e+308 (finite: True)
ujson:        1.7976931345e+308 -> inf (finite: False)
```

## Why This Is A Bug

1. **Round-trip property violated**: `ujson_loads(ujson_dumps(x)) != x` for large but valid floats
2. **Silent data corruption**: Finite values become infinity without warning
3. **Inconsistent with stdlib**: Python's `json` module handles these values correctly
4. **Default precision insufficient**: The default `double_precision=10` is too low for edge cases near max float

## Fix

The issue is that `double_precision=10` (the default) loses precision for large floats. The serialized form `"1.797693135e+308"` overflows when parsed back.

Increasing precision to 15 fixes the issue:

```python
from pandas.io.json import ujson_dumps, ujson_loads

value = 1.7976931345e+308
result = ujson_loads(ujson_dumps(value, double_precision=15))
print(f"{value} -> {result}")  # 1.7976931345e+308 -> 1.7976931345e+308
```

**Recommended fix**: Change the default `double_precision` from 10 to 15 (the maximum supported value) to match float64 precision and prevent such overflow issues.

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -142,7 +142,7 @@ def to_json(
     obj: NDFrame,
     orient: str | None = None,
     date_format: str = "epoch",
-    double_precision: int = 10,
+    double_precision: int = 15,
     force_ascii: bool = True,
     date_unit: str = "ms",
     default_handler: Callable[[Any], JSONSerializable] | None = None,
```

This would align ujson's behavior more closely with the standard library's `json.dumps()` which uses full precision by default.