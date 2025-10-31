# Bug Report: xarray.compat.array_api_compat.result_type crashes with string and bytes scalars

**Target**: `xarray.compat.array_api_compat.result_type`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `result_type()` function crashes when given string or bytes scalar values because it incorrectly delegates to `np.result_type()`, which interprets strings as dtype format strings rather than as weak scalar values that the xarray infrastructure is designed to handle.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Hypothesis-based property test demonstrating the xarray.compat.array_api_compat.result_type
crash with string and bytes scalar inputs.
"""
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import given, strategies as st, settings
from xarray.compat.array_api_compat import result_type


@given(st.text(max_size=10))
@settings(max_examples=100)
def test_result_type_with_str_scalar(value):
    """Test that result_type can handle string scalars."""
    result = result_type(value, xp=np)
    assert isinstance(result, np.dtype)


@given(st.binary(max_size=10))
@settings(max_examples=100)
def test_result_type_with_bytes_scalar(value):
    """Test that result_type can handle bytes scalars."""
    result = result_type(value, xp=np)
    assert isinstance(result, np.dtype)


if __name__ == "__main__":
    print("Running property-based tests for result_type with string/bytes scalars...")
    print("\nTesting with string scalars:")
    try:
        test_result_type_with_str_scalar()
        print("  ✓ All string tests passed")
    except Exception as e:
        print(f"  ✗ String test failed: {e}")

    print("\nTesting with bytes scalars:")
    try:
        test_result_type_with_bytes_scalar()
        print("  ✓ All bytes tests passed")
    except Exception as e:
        print(f"  ✗ Bytes test failed: {e}")
```

<details>

<summary>
**Failing input**: `''` (empty string) and `b''` (empty bytes)
</summary>
```
Falsifying example: test_result_type_with_str_scalar(
    value='',
)
Traceback (most recent call last):
  File "hypo.py", line 13, in test_result_type_with_str_scalar
    result = result_type(value, xp=np)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/compat/array_api_compat.py", line 44, in result_type
    return xp.result_type(*arrays_and_dtypes)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: data type '' not understood

Falsifying example: test_result_type_with_bytes_scalar(
    value=b'',
)
Traceback (most recent call last):
  File "hypo.py", line 18, in test_result_type_with_bytes_scalar
    result = result_type(value, xp=np)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/compat/array_api_compat.py", line 44, in result_type
    return xp.result_type(*arrays_and_dtypes)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: data type '' not understood
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal demonstration of xarray.compat.array_api_compat.result_type crash
with string and bytes scalar inputs.
"""
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from xarray.compat.array_api_compat import result_type

# Test case 1: Empty string
print("Test 1: result_type('', xp=np)")
try:
    result = result_type('', xp=np)
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Test case 2: String '0:'
print("\nTest 2: result_type('0:', xp=np)")
try:
    result = result_type('0:', xp=np)
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Test case 3: String '01'
print("\nTest 3: result_type('01', xp=np)")
try:
    result = result_type('01', xp=np)
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Test case 4: Empty bytes
print("\nTest 4: result_type(b'', xp=np)")
try:
    result = result_type(b'', xp=np)
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Test case 5: String 'test'
print("\nTest 5: result_type('test', xp=np)")
try:
    result = result_type('test', xp=np)
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Test case 6: Bytes b'test'
print("\nTest 6: result_type(b'test', xp=np)")
try:
    result = result_type(b'test', xp=np)
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")
```

<details>

<summary>
Crashes with TypeError/ValueError/SyntaxError depending on string content
</summary>
```
Test 1: result_type('', xp=np)
  Error: TypeError: data type '' not understood

Test 2: result_type('0:', xp=np)
  Error: ValueError: format number 1 of "0:" is not recognized

Test 3: result_type('01', xp=np)
  Error: SyntaxError: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers (<unknown>, line 1)

Test 4: result_type(b'', xp=np)
  Error: TypeError: data type '' not understood

Test 5: result_type('test', xp=np)
  Error: TypeError: data type 'test' not understood

Test 6: result_type(b'test', xp=np)
  Error: TypeError: data type 'test' not understood
```
</details>

## Why This Is A Bug

The `result_type()` function contains a logic error that causes it to crash on string and bytes scalar inputs, even though the xarray codebase explicitly includes infrastructure to handle these cases correctly.

The codebase defines strings and bytes as "weak scalar types" through the `is_weak_scalar_type()` helper function (line 7 in array_api_compat.py), which explicitly checks for `str | bytes`. Furthermore, there's a complete implementation in `_future_array_api_result_type()` (lines 10-37) that correctly handles these weak scalar types by converting them to appropriate NumPy dtypes (`str` → Unicode string dtype, `bytes` → bytes dtype).

However, when `xp is np` (line 41), the current implementation takes a shortcut and directly calls `np.result_type(*arrays_and_dtypes)` without checking if any arguments are strings or bytes. NumPy's `result_type()` interprets string arguments as dtype format strings (e.g., 'i4' for int32, 'f8' for float64) rather than as scalar values. When given arbitrary strings that aren't valid dtype format strings, NumPy raises various errors.

The bug contradicts the clear intent in the code, which includes a comment referencing GitHub issue #805 on the Array API repository about extending result_type to handle Python scalars. The infrastructure exists and works correctly when called directly - `_future_array_api_result_type('', xp=np)` returns `dtype('<U1')` as expected.

## Relevant Context

The xarray implementation appears to be anticipating a future Array API specification that would clarify scalar handling. The comment at line 11-13 states: "fallback implementation for `xp.result_type` with python scalars. Can be removed once a version of the Array API that includes https://github.com/data-apis/array-api/issues/805 can be required."

The `_future_array_api_result_type()` function successfully handles these cases:
- `_future_array_api_result_type('', xp=np)` returns `dtype('<U1')` (Unicode string)
- `_future_array_api_result_type(b'', xp=np)` returns `dtype('|S1')` (bytes)

NumPy's documented behavior for `result_type()` is to interpret string arguments as dtype format strings, not as scalar values. This is why strings like 'i4', 'f8', 'c16' work (they're valid dtype strings), while arbitrary strings fail.

The Array API specification mentions that `result_type()` accepts "an arbitrary number of input arrays, scalars, and/or dtypes" but doesn't explicitly define whether strings and bytes count as scalars or how they should be promoted.

## Proposed Fix

```diff
--- a/xarray/compat/array_api_compat.py
+++ b/xarray/compat/array_api_compat.py
@@ -38,7 +38,10 @@ def _future_array_api_result_type(*arrays_and_dtypes, xp):


 def result_type(*arrays_and_dtypes, xp) -> np.dtype:
-    if xp is np or any(
+    # Check if any arguments are weak scalars that np.result_type can't handle
+    has_str_or_bytes = any(isinstance(t, (str, bytes)) for t in arrays_and_dtypes)
+
+    if not has_str_or_bytes and (xp is np or any(
         isinstance(getattr(t, "dtype", t), np.dtype) for t in arrays_and_dtypes
-    ):
+    )):
         return xp.result_type(*arrays_and_dtypes)
     else:
         return _future_array_api_result_type(*arrays_and_dtypes, xp=xp)
```