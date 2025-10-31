# Bug Report: pandas.io.parsers._stringify_na_values Float Type Lost

**Target**: `pandas.io.parsers.readers._stringify_na_values`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_stringify_na_values` function incorrectly loses float type information when `floatify=True` and the input is an integer-valued float (e.g., `5.0`, `"5.0"`). The function mutates a local variable `v` from float to int, causing the wrong type to be added to the result set.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pytest


@given(integer_as_float=st.integers(min_value=-1000, max_value=1000).map(float))
def test_stringify_na_values_preserves_float_type(integer_as_float):
    """
    Property: When floatify=True, _stringify_na_values should include
    float versions of numeric values in the result set.
    """
    from pandas.io.parsers.readers import _stringify_na_values

    result = _stringify_na_values([str(integer_as_float)], floatify=True)

    assert isinstance(result, set)
    has_float = any(isinstance(item, float) and item == integer_as_float for item in result)
    assert has_float, f"Expected float {integer_as_float} in result when floatify=True"
```

**Failing input**: `"5.0"` with `floatify=True`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.io.parsers.readers import _stringify_na_values

result = _stringify_na_values(["5.0"], floatify=True)

print(f"Result: {result}")
for item in sorted(result, key=str):
    print(f"  {repr(item):10} -> {type(item).__name__}")

print(f"\nBug: float 5.0 is NOT in result")
print(f"  5.0 in result: {5.0 in result}")
print(f"  5 in result: {5 in result}")
```

Output:
```
Result: {5, '5', '5.0'}
  5          -> int
  '5'        -> str
  '5.0'      -> str

Bug: float 5.0 is NOT in result
  5.0 in result: False
  5 in result: True
```

## Why This Is A Bug

The function's docstring states it should "return a stringified and numeric for these values". When `floatify=True`, the intent is to include numeric (float) representations in addition to string representations. However, due to a variable reassignment bug on line 2117, the float type is lost for integer-valued inputs.

The problematic code flow:
1. Line 2113: `v = float(x)` - creates a float
2. Line 2116-2117: `if v == int(v): v = int(v)` - **reassigns v to int**
3. Line 2121-2122: `if floatify: result.append(v)` - appends the **int** instead of the float

This means NA values like `"5.0"` will only match against integer `5` in the resulting set, not float `5.0`, which could cause incorrect NA value detection in edge cases.

## Fix

```diff
--- a/pandas/io/parsers/readers.py
+++ b/pandas/io/parsers/readers.py
@@ -2113,12 +2113,14 @@ def _stringify_na_values(na_values, floatify: bool):
         try:
             v = float(x)

+            # Store original float before potentially converting to int
+            float_v = v
+
             # we are like 999 here
             if v == int(v):
                 v = int(v)
                 result.append(f"{v}.0")
                 result.append(str(v))

             if floatify:
-                result.append(v)
+                result.append(float_v)
         except (TypeError, ValueError, OverflowError):
             pass
```

Alternatively, avoid the mutation entirely:

```diff
--- a/pandas/io/parsers/readers.py
+++ b/pandas/io/parsers/readers.py
@@ -2114,11 +2114,11 @@ def _stringify_na_values(na_values, floatify: bool):
             v = float(x)

             # we are like 999 here
             if v == int(v):
-                v = int(v)
-                result.append(f"{v}.0")
-                result.append(str(v))
+                int_v = int(v)
+                result.append(f"{int_v}.0")
+                result.append(str(int_v))

             if floatify:
                 result.append(v)
```