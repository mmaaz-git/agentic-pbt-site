# Bug Report: attrs.converters.to_bool Accepts Undocumented Float Values

**Target**: `attrs.converters.to_bool`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `attrs.converters.to_bool()` function accepts float values `1.0` and `0.0` despite documentation explicitly stating it only accepts specific boolean, string, and integer values, violating its documented contract.

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

if __name__ == "__main__":
    print("Testing that to_bool rejects most floats but accepts 1.0 and 0.0...")
    print()

    print("Test 1: to_bool should reject floats other than 1.0 and 0.0")
    test_to_bool_rejects_undocumented_floats()
    print("✓ Passed - all floats except 1.0 and 0.0 are rejected")
    print()

    print("Test 2: to_bool accepts 1.0 and 0.0 (undocumented behavior)")
    test_to_bool_accepts_some_floats()
    print("✓ Passed - 1.0 converts to True and 0.0 converts to False")
    print()

    print("This demonstrates the bug: 1.0 and 0.0 are accepted despite not being")
    print("in the documented list of accepted values.")
```

<details>

<summary>
**Failing input**: `1.0` and `0.0` (accepted despite not being documented)
</summary>
```
Testing that to_bool rejects most floats but accepts 1.0 and 0.0...

Test 1: to_bool should reject floats other than 1.0 and 0.0
✓ Passed - all floats except 1.0 and 0.0 are rejected

Test 2: to_bool accepts 1.0 and 0.0 (undocumented behavior)
✓ Passed - 1.0 converts to True and 0.0 converts to False

This demonstrates the bug: 1.0 and 0.0 are accepted despite not being
in the documented list of accepted values.
```
</details>

## Reproducing the Bug

```python
from attrs import converters

print("Documented behavior:")
print(f"to_bool(1) = {converters.to_bool(1)}")
print(f"to_bool(0) = {converters.to_bool(0)}")
print(f"to_bool(True) = {converters.to_bool(True)}")
print(f"to_bool(False) = {converters.to_bool(False)}")
print(f'to_bool("1") = {converters.to_bool("1")}')
print(f'to_bool("0") = {converters.to_bool("0")}')

print("\nUndocumented behavior - accepts these floats:")
print(f"to_bool(1.0) = {converters.to_bool(1.0)}")
print(f"to_bool(0.0) = {converters.to_bool(0.0)}")

print("\nInconsistent behavior - rejects these floats:")
try:
    result = converters.to_bool(1.5)
    print(f"to_bool(1.5) = {result}")
except ValueError as e:
    print(f"to_bool(1.5) raises ValueError: {e}")

try:
    result = converters.to_bool(2.0)
    print(f"to_bool(2.0) = {result}")
except ValueError as e:
    print(f"to_bool(2.0) raises ValueError: {e}")

try:
    result = converters.to_bool(0.5)
    print(f"to_bool(0.5) = {result}")
except ValueError as e:
    print(f"to_bool(0.5) raises ValueError: {e}")

try:
    result = converters.to_bool(-1.0)
    print(f"to_bool(-1.0) = {result}")
except ValueError as e:
    print(f"to_bool(-1.0) raises ValueError: {e}")
```

<details>

<summary>
Float values 1.0 and 0.0 are accepted, while other floats are rejected
</summary>
```
Documented behavior:
to_bool(1) = True
to_bool(0) = False
to_bool(True) = True
to_bool(False) = False
to_bool("1") = True
to_bool("0") = False

Undocumented behavior - accepts these floats:
to_bool(1.0) = True
to_bool(0.0) = False

Inconsistent behavior - rejects these floats:
to_bool(1.5) raises ValueError: Cannot convert value to bool: 1.5
to_bool(2.0) raises ValueError: Cannot convert value to bool: 2.0
to_bool(0.5) raises ValueError: Cannot convert value to bool: 0.5
to_bool(-1.0) raises ValueError: Cannot convert value to bool: -1.0
```
</details>

## Why This Is A Bug

The `to_bool` converter's documentation in `/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages/attr/converters.py:126-151` explicitly lists all accepted values:

- For True: `True`, `"true"`, `"t"`, `"yes"`, `"y"`, `"on"`, `"1"`, `1`
- For False: `False`, `"false"`, `"f"`, `"no"`, `"n"`, `"off"`, `"0"`, `0`
- Line 149 states: "Raises: ValueError: For any other value."

Float values `1.0` and `0.0` are not in these lists, yet they are accepted. This happens because Python's equality operator treats `1.0 == 1` and `0.0 == 0` as `True`. The implementation at lines 156-159 uses membership testing with `in`:

```python
if val in (True, "true", "t", "yes", "y", "on", "1", 1):
    return True
if val in (False, "false", "f", "no", "n", "off", "0", 0):
    return False
```

This creates three problems:

1. **Contract violation**: The function accepts values not listed in its documentation, violating the explicit contract that "any other value" should raise ValueError
2. **Inconsistent behavior**: Accepts `1.0` and `0.0` but rejects other floats like `1.5`, `2.0`, creating surprising and inconsistent behavior
3. **Type confusion**: A boolean converter accepting float values may lead to unexpected behavior in type-sensitive code

## Relevant Context

The `to_bool` converter was added in attrs version 21.3.0 and is designed primarily to convert string representations (e.g., from environment variables) and integer flags to boolean values. The documentation at https://www.attrs.org/en/stable/api.html#converters confirms the same list of accepted values.

The bug occurs due to Python's duck typing where numeric types can be compared for equality across types. While `1.0 == 1` is standard Python behavior, the converter's documentation promises strict value checking that should reject undocumented inputs.

This edge case is unlikely to affect many users but represents a clear deviation from the documented behavior. The fix would make the implementation match the documentation exactly.

## Proposed Fix

```diff
--- a/attr/converters.py
+++ b/attr/converters.py
@@ -150,6 +150,10 @@ def to_bool(val):

     .. versionadded:: 21.3.0
     """
+    # Explicitly reject float values to match documentation
+    if isinstance(val, float):
+        msg = f"Cannot convert value to bool: {val!r}"
+        raise ValueError(msg)
+
     if isinstance(val, str):
         val = val.lower()

```