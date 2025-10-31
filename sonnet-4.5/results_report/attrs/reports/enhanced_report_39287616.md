# Bug Report: attrs.converters.to_bool Undocumented Float Acceptance

**Target**: `attrs.converters.to_bool`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `attrs.converters.to_bool()` function incorrectly accepts float values 0.0 and 1.0, violating its documented contract which explicitly lists only integers 0 and 1 as valid numeric inputs.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test demonstrating attrs.converters.to_bool float acceptance bug"""

from hypothesis import given, strategies as st, settings
from attrs import converters

@given(st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=500)
def test_to_bool_rejects_floats(val):
    """Test that to_bool rejects all float values as per documentation."""
    try:
        result = converters.to_bool(val)
        assert False, f"to_bool({val!r}) should raise ValueError but returned {result}"
    except ValueError:
        pass  # This is the expected behavior for all floats

if __name__ == "__main__":
    # Run the test
    print("Running property-based test for attrs.converters.to_bool...")
    print("=" * 60)
    try:
        test_to_bool_rejects_floats()
        print("Test passed! All float values correctly raised ValueError.")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nThis demonstrates that to_bool accepts some float values")
        print("despite the documentation only listing integers 0 and 1.")
```

<details>

<summary>
**Failing input**: `0.0`
</summary>
```
Running property-based test for attrs.converters.to_bool...
============================================================
Test failed: to_bool(0.0) should raise ValueError but returned False

This demonstrates that to_bool accepts some float values
despite the documentation only listing integers 0 and 1.
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of attrs.converters.to_bool float acceptance bug"""

from attrs import converters

# Test cases that should raise ValueError according to documentation
# but actually succeed (BUG)
print("Testing float values that should raise ValueError:")
print("-" * 50)

try:
    result = converters.to_bool(0.0)
    print(f"converters.to_bool(0.0) = {result} (BUG: Should raise ValueError)")
except ValueError as e:
    print(f"converters.to_bool(0.0) raised ValueError: {e}")

try:
    result = converters.to_bool(1.0)
    print(f"converters.to_bool(1.0) = {result} (BUG: Should raise ValueError)")
except ValueError as e:
    print(f"converters.to_bool(1.0) raised ValueError: {e}")

print("\n" + "=" * 50 + "\n")

# Test other float values to show inconsistency
print("Testing other float values (correctly raise ValueError):")
print("-" * 50)

test_values = [0.5, 1.5, 2.0, -1.0, 10.0]
for val in test_values:
    try:
        result = converters.to_bool(val)
        print(f"converters.to_bool({val}) = {result} (BUG: Should raise ValueError)")
    except ValueError as e:
        print(f"converters.to_bool({val}) correctly raised ValueError")

print("\n" + "=" * 50 + "\n")

# Demonstrate why this happens
print("Root cause - Python's equality behavior:")
print("-" * 50)
print(f"0.0 == 0: {0.0 == 0}")
print(f"1.0 == 1: {1.0 == 1}")
print(f"0.0 in (0,): {0.0 in (0,)}")
print(f"1.0 in (1,): {1.0 in (1,)}")

print("\n" + "=" * 50 + "\n")

# Show documented valid inputs for comparison
print("Documented valid inputs (from docstring):")
print("-" * 50)
print("Values mapping to True: True, 'true'/'t', 'yes'/'y', 'on', '1', 1")
print("Values mapping to False: False, 'false'/'f', 'no'/'n', 'off', '0', 0")
print("\nNote: The documentation lists integer 1 and 0, NOT float 1.0 and 0.0")
```

<details>

<summary>
converters.to_bool accepts floats 0.0 and 1.0 but rejects all other float values
</summary>
```
Testing float values that should raise ValueError:
--------------------------------------------------
converters.to_bool(0.0) = False (BUG: Should raise ValueError)
converters.to_bool(1.0) = True (BUG: Should raise ValueError)

==================================================

Testing other float values (correctly raise ValueError):
--------------------------------------------------
converters.to_bool(0.5) correctly raised ValueError
converters.to_bool(1.5) correctly raised ValueError
converters.to_bool(2.0) correctly raised ValueError
converters.to_bool(-1.0) correctly raised ValueError
converters.to_bool(10.0) correctly raised ValueError

==================================================

Root cause - Python's equality behavior:
--------------------------------------------------
0.0 == 0: True
1.0 == 1: True
0.0 in (0,): True
1.0 in (1,): True

==================================================

Documented valid inputs (from docstring):
--------------------------------------------------
Values mapping to True: True, 'true'/'t', 'yes'/'y', 'on', '1', 1
Values mapping to False: False, 'false'/'f', 'no'/'n', 'off', '0', 0

Note: The documentation lists integer 1 and 0, NOT float 1.0 and 0.0
```
</details>

## Why This Is A Bug

This violates the documented API contract in several ways:

1. **Explicit documentation**: The docstring at `/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages/attr/converters.py:125-151` explicitly lists the valid values:
   - For True: `True`, `"true"`/`"t"`, `"yes"`/`"y"`, `"on"`, `"1"`, `1` (integer)
   - For False: `False`, `"false"`/`"f"`, `"no"`/`"n"`, `"off"`, `"0"`, `0` (integer)

   The documentation specifically shows integer literals `0` and `1`, not float literals `0.0` and `1.0`.

2. **Inconsistent behavior**: The function rejects all other float values (0.5, 1.5, 2.0, etc.) with ValueError, but accepts 0.0 and 1.0. This inconsistency suggests unintended behavior rather than deliberate design.

3. **Type safety violation**: Users relying on the documented contract for strict type validation may encounter unexpected behavior when float values pass through without raising exceptions.

4. **Implementation detail leaking**: The bug occurs because Python's `in` operator uses equality checking, where `0.0 == 0` and `1.0 == 1` evaluate to `True`. This is an implementation detail that shouldn't affect the public API contract.

## Relevant Context

The bug stems from the implementation at lines 156-159 of `/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages/attr/converters.py`:

```python
if val in (True, "true", "t", "yes", "y", "on", "1", 1):
    return True
if val in (False, "false", "f", "no", "n", "off", "0", 0):
    return False
```

The `in` operator uses Python's equality semantics, where:
- `0.0 in (0,)` returns `True` because `0.0 == 0`
- `1.0 in (1,)` returns `True` because `1.0 == 1`

This is a common pitfall in Python when mixing numeric types in membership tests. The attrs library documentation is available at https://www.attrs.org/en/stable/api.html#converters and confirms the same values as the inline documentation.

## Proposed Fix

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
+    # Check membership and ensure type matches documented types (bool, str, int)
+    if val in truthy and type(val) in (bool, str, int):
         return True
-    if val in (False, "false", "f", "no", "n", "off", "0", 0):
+    if val in falsy and type(val) in (bool, str, int):
         return False

     msg = f"Cannot convert value to bool: {val!r}"
     raise ValueError(msg)
```