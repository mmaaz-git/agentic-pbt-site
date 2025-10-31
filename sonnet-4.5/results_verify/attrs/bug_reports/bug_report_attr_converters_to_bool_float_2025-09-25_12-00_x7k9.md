# Bug Report: attr.converters.to_bool Accepts Undocumented Float Values

**Target**: `attr.converters.to_bool`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `attr.converters.to_bool` function accepts float values `1.0` and `0.0` and converts them to `True` and `False` respectively, even though the documentation explicitly states that only integers `1` and `0` are valid numeric inputs.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import attr

@given(st.floats())
def test_to_bool_only_documented_types(val):
    """to_bool should only accept documented types: bool, str, and specific ints."""
    documented_values = {
        True, False,
        "true", "t", "yes", "y", "on", "1",
        "false", "f", "no", "n", "off", "0",
        1, 0
    }

    if val not in documented_values and val not in {1.0, 0.0}:
        with pytest.raises(ValueError):
            attr.converters.to_bool(val)

    if val in {1.0, 0.0} and val not in documented_values:
        with pytest.raises(ValueError):
            attr.converters.to_bool(val)
```

**Failing input**: `1.0` and `0.0`

## Reproducing the Bug

```python
import attr

result_1 = attr.converters.to_bool(1.0)
print(f"to_bool(1.0) = {result_1}")

result_0 = attr.converters.to_bool(0.0)
print(f"to_bool(0.0) = {result_0}")
```

**Output:**
```
to_bool(1.0) = True
to_bool(0.0) = False
```

## Why This Is A Bug

According to the docstring in `attr/converters.py` lines 130-151, `to_bool` documents these mappings:

**Values mapping to True:**
- `True`
- `"true"` / `"t"`
- `"yes"` / `"y"`
- `"on"`
- `"1"`
- `1` (integer)

**Values mapping to False:**
- `False`
- `"false"` / `"f"`
- `"no"` / `"n"`
- `"off"`
- `"0"`
- `0` (integer)

Notice that **floats are not mentioned** in the documentation. However, the implementation uses Python's `in` operator with tuple membership testing (lines 156-159):

```python
if val in (True, "true", "t", "yes", "y", "on", "1", 1):
    return True
if val in (False, "false", "f", "no", "n", "off", "0", 0):
    return False
```

Since Python's `in` operator uses `==` for equality comparison, and `1.0 == 1` and `0.0 == 0` both evaluate to `True`, the function inadvertently accepts these float values. This violates the documented API contract.

## Fix

```diff
--- a/attr/converters.py
+++ b/attr/converters.py
@@ -151,6 +151,10 @@ def to_bool(val):

     .. versionadded:: 21.3.0
     """
+    if not isinstance(val, (bool, str, int)):
+        msg = f"Cannot convert value to bool: {val!r}"
+        raise ValueError(msg)
+
     if isinstance(val, str):
         val = val.lower()

```

Alternatively, if accepting floats equal to 0.0 or 1.0 is intentional, the documentation should be updated to reflect this behavior.