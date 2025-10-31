# Bug Report: attrs to_bool Undocumented Float Acceptance

**Target**: `attrs.converters.to_bool`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `attrs.converters.to_bool()` converter accepts `1.0` and `0.0` (floats) despite documentation stating it only accepts specific boolean values, creating undocumented and inconsistent behavior.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import attrs
from attrs import converters

@given(st.floats().filter(lambda x: x not in [1.0, 0.0]))
def test_to_bool_rejects_undocumented_floats(x):
    try:
        converters.to_bool(x)
        assert False, f"to_bool should reject float {x}"
    except ValueError:
        pass

@given(st.sampled_from([1.0, 0.0]))
def test_to_bool_accepts_some_floats(x):
    result = converters.to_bool(x)
    assert result == (x == 1.0)
```

**Failing input**: `1.0` and `0.0` are accepted despite not being documented

## Reproducing the Bug

```python
from attrs import converters

print("Documented behavior:")
print(f"to_bool(1) = {converters.to_bool(1)}")
print(f"to_bool(0) = {converters.to_bool(0)}")

print("\nUndocumented behavior:")
print(f"to_bool(1.0) = {converters.to_bool(1.0)}")
print(f"to_bool(0.0) = {converters.to_bool(0.0)}")

print("\nInconsistent behavior:")
try:
    print(f"to_bool(1.5) = {converters.to_bool(1.5)}")
except ValueError:
    print("to_bool(1.5) raises ValueError")

try:
    print(f"to_bool(2.0) = {converters.to_bool(2.0)}")
except ValueError:
    print("to_bool(2.0) raises ValueError")
```

## Why This Is A Bug

The `to_bool` converter documentation explicitly lists the values it accepts:
- For True: `True`, `"true"`, `"t"`, `"yes"`, `"y"`, `"on"`, `"1"`, `1`
- For False: `False`, `"false"`, `"f"`, `"no"`, `"n"`, `"off"`, `"0"`, `0`
- States: "Raises ValueError: For any other value."

However, due to Python's numeric equality (`1.0 == 1` and `0.0 == 0`), the implementation at line 156-159 in `/attr/converters.py`:

```python
if val in (True, "true", "t", "yes", "y", "on", "1", 1):
    return True
if val in (False, "false", "f", "no", "n", "off", "0", 0):
    return False
```

This creates three problems:
1. **Contract violation**: Accepts values not listed in documentation
2. **Inconsistent behavior**: Accepts `1.0` and `0.0` but rejects `1.5`, `2.0`, etc.
3. **Surprising behavior**: Users may not expect float values to be accepted

## Fix

The fix should explicitly check for float type and reject all floats to match the documentation:

```diff
--- a/attr/converters.py
+++ b/attr/converters.py
@@ -150,6 +150,10 @@ def to_bool(val):

     .. versionadded:: 21.3.0
     """
+    if isinstance(val, float):
+        msg = f"Cannot convert value to bool: {val!r}"
+        raise ValueError(msg)
+
     if isinstance(val, str):
         val = val.lower()

@@ -157,6 +161,7 @@ def to_bool(val):
         return True
     if val in (False, "false", "f", "no", "n", "off", "0", 0):
         return False

     msg = f"Cannot convert value to bool: {val!r}"
     raise ValueError(msg)
```

Alternatively, if float support is desired, the documentation should be updated to explicitly list `1.0` and `0.0` as accepted values.