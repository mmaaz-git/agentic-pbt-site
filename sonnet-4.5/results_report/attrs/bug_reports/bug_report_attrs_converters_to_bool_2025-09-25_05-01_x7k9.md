# Bug Report: attrs converters.to_bool Float Acceptance

**Target**: `attrs.converters.to_bool`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `attrs.converters.to_bool()` function accepts float values `0.0` and `1.0` despite documentation explicitly listing only integers `0` and `1` as valid numeric inputs.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from attrs import converters

@given(st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=500)
def test_to_bool_rejects_floats(val):
    try:
        result = converters.to_bool(val)
        assert False, f"to_bool({val!r}) should raise ValueError but returned {result}"
    except ValueError:
        pass
```

**Failing input**: `0.0` (returns `False` instead of raising `ValueError`)

## Reproducing the Bug

```python
from attrs import converters

result_0 = converters.to_bool(0.0)
result_1 = converters.to_bool(1.0)

print(f"to_bool(0.0) = {result_0}")
print(f"to_bool(1.0) = {result_1}")
```

Output:
```
to_bool(0.0) = False
to_bool(1.0) = True
```

Expected: Both should raise `ValueError` since floats are not documented as valid inputs.

## Why This Is A Bug

The documentation in `attr/converters.py` explicitly states:

> Values mapping to `True`: `True`, `"true"` / `"t"`, `"yes"` / `"y"`, `"on"`, `"1"`, `1`
>
> Values mapping to `False`: `False`, `"false"` / `"f"`, `"no"` / `"n"`, `"off"`, `"0"`, `0`

Note that the numeric values listed are `0` and `1` (integers), not `0.0` and `1.0` (floats).

The bug occurs because:
1. The implementation uses `in` checks on lines 156 and 158
2. Python's equality operator treats `0.0 == 0` and `1.0 == 1` as `True`
3. Therefore `0.0 in (..., 0)` and `1.0 in (..., 1)` both return `True`
4. This causes undocumented float values to be accepted

This violates the documented API contract and may surprise users who expect strict type checking based on the documentation.

## Fix

```diff
--- a/attr/converters.py
+++ b/attr/converters.py
@@ -153,10 +153,14 @@ def to_bool(val):
     if isinstance(val, str):
         val = val.lower()

-    if val in (True, "true", "t", "yes", "y", "on", "1", 1):
+    truthy = (True, "true", "t", "yes", "y", "on", "1", 1)
+    falsy = (False, "false", "f", "no", "n", "off", "0", 0)
+
+    if val in truthy and type(val) in (bool, str, int):
         return True
-    if val in (False, "false", "f", "no", "n", "off", "0", 0):
+    if val in falsy and type(val) in (bool, str, int):
         return False

     msg = f"Cannot convert value to bool: {val!r}"
     raise ValueError(msg)
```

This fix ensures that only the documented types (bool, str, int) are accepted, rejecting floats like `0.0` and `1.0`.