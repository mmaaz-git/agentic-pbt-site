# Bug Report: pandas.api.types.infer_dtype Crashes on Python Built-in Numeric Scalars

**Target**: `pandas.api.types.infer_dtype`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `infer_dtype` function crashes with `TypeError: 'X' object is not iterable` when given Python built-in scalar types (int, float, bool, complex, None), despite being documented to accept "scalar" values and working correctly for other scalar types like strings and NumPy scalars.

## Property-Based Test

```python
import pandas.api.types as types
from hypothesis import given, strategies as st


@given(st.one_of(
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
))
def test_infer_dtype_accepts_scalars(val):
    result_scalar = types.infer_dtype(val, skipna=False)
    result_list = types.infer_dtype([val], skipna=False)
    assert result_scalar == result_list

if __name__ == "__main__":
    test_infer_dtype_accepts_scalars()
```

<details>

<summary>
**Failing input**: `0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 16, in <module>
    test_infer_dtype_accepts_scalars()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 6, in test_infer_dtype_accepts_scalars
    st.integers(),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 11, in test_infer_dtype_accepts_scalars
    result_scalar = types.infer_dtype(val, skipna=False)
  File "pandas/_libs/lib.pyx", line 1621, in pandas._libs.lib.infer_dtype
TypeError: 'int' object is not iterable
Falsifying example: test_infer_dtype_accepts_scalars(
    val=0,
)
```
</details>

## Reproducing the Bug

```python
"""
Demonstrating the crash of pandas.api.types.infer_dtype with Python scalar types.
"""

import pandas.api.types as types
import numpy as np

print("Testing pandas.api.types.infer_dtype with various scalar types\n")
print("=" * 60)

# Test cases that should work according to documentation but crash
test_cases_crash = [
    ("Python int", 0),
    ("Python float", 1.5),
    ("Python bool", True),
    ("Python complex", 1+2j),
    ("None", None)
]

print("\n1. PYTHON BUILT-IN SCALARS (These crash):\n")
for name, value in test_cases_crash:
    try:
        result = types.infer_dtype(value, skipna=False)
        print(f"{name:20} value={str(value):10} → {result}")
    except TypeError as e:
        print(f"{name:20} value={str(value):10} → ERROR: {e}")

print("\n" + "=" * 60)
print("\n2. SCALAR TYPES THAT WORK:\n")

# Test cases that work
test_cases_work = [
    ("String", "hello"),
    ("Bytes", b"bytes"),
    ("NumPy int64", np.int64(5)),
    ("NumPy float64", np.float64(5.5))
]

for name, value in test_cases_work:
    try:
        result = types.infer_dtype(value, skipna=False)
        print(f"{name:20} value={value!r:15} → {result}")
    except TypeError as e:
        print(f"{name:20} value={value!r:15} → ERROR: {e}")

print("\n" + "=" * 60)
print("\n3. THE SAME VALUES WRAPPED IN LISTS (All work):\n")

# Test that same values work when wrapped in lists
for name, value in test_cases_crash:
    try:
        result = types.infer_dtype([value], skipna=False)
        print(f"{name:20} value=[{value}] → {result}")
    except TypeError as e:
        print(f"{name:20} value=[{value}] → ERROR: {e}")

print("\n" + "=" * 60)
print("\n4. VERIFYING SCALAR STATUS WITH pandas.api.types.is_scalar:\n")

# Verify that pandas considers these as scalars
for name, value in test_cases_crash + test_cases_work:
    is_scalar = types.is_scalar(value)
    print(f"{name:20} is_scalar={is_scalar}")

print("\n" + "=" * 60)
print("\nCONCLUSION:")
print("The function is documented to accept 'scalar' values but crashes on")
print("common Python scalar types (int, float, bool, complex, None) while")
print("working for some other scalars (str, bytes, numpy scalars).")
```

<details>

<summary>
TypeError: 'int' object is not iterable (and similar for float, bool, complex, NoneType)
</summary>
```
Testing pandas.api.types.infer_dtype with various scalar types

============================================================

1. PYTHON BUILT-IN SCALARS (These crash):

Python int           value=0          → ERROR: 'int' object is not iterable
Python float         value=1.5        → ERROR: 'float' object is not iterable
Python bool          value=True       → ERROR: 'bool' object is not iterable
Python complex       value=(1+2j)     → ERROR: 'complex' object is not iterable
None                 value=None       → ERROR: 'NoneType' object is not iterable

============================================================

2. SCALAR TYPES THAT WORK:

String               value='hello'         → string
Bytes                value=b'bytes'        → integer
NumPy int64          value=np.int64(5)     → integer
NumPy float64        value=np.float64(5.5) → floating

============================================================

3. THE SAME VALUES WRAPPED IN LISTS (All work):

Python int           value=[0] → integer
Python float         value=[1.5] → floating
Python bool          value=[True] → boolean
Python complex       value=[(1+2j)] → complex
None                 value=[None] → mixed

============================================================

4. VERIFYING SCALAR STATUS WITH pandas.api.types.is_scalar:

Python int           is_scalar=True
Python float         is_scalar=True
Python bool          is_scalar=True
Python complex       is_scalar=True
None                 is_scalar=True
String               is_scalar=True
Bytes                is_scalar=True
NumPy int64          is_scalar=True
NumPy float64        is_scalar=True

============================================================

CONCLUSION:
The function is documented to accept 'scalar' values but crashes on
common Python scalar types (int, float, bool, complex, None) while
working for some other scalars (str, bytes, numpy scalars).
```
</details>

## Why This Is A Bug

This violates the documented API contract in several critical ways:

1. **Documentation explicitly promises scalar support**: The function's docstring states "Return a string label of the type of a scalar or list-like of values" with the parameter documented as "scalar, list, ndarray, or pandas type". The word "scalar" is listed first, indicating it's a primary use case.

2. **Inconsistent behavior across scalar types**: The function accepts some scalars (strings like "hello", bytes like b"bytes", NumPy scalars like np.int64(5)) but rejects the most fundamental Python scalar types (int, float, bool, complex, None). This inconsistency is confusing and unpredictable.

3. **Pandas itself recognizes these as scalars**: Using `pandas.api.types.is_scalar()`, all the failing inputs return `True`, meaning pandas officially considers them scalars. The function fails on values that pandas' own type system identifies as scalars.

4. **The workaround reveals the bug**: When these exact same values are wrapped in a list, the function works perfectly and returns the correct type inference. This proves the function can handle these types, but has a bug in its scalar input handling.

5. **Error message indicates implementation bug**: The error "TypeError: 'int' object is not iterable" shows the function is trying to iterate over the input without first checking if it's a scalar that needs to be wrapped.

## Relevant Context

- **Pandas version tested**: 2.3.2
- **Function location**: `pandas._libs.lib.infer_dtype` (implemented in Cython at pandas/_libs/lib.pyx:1621)
- **Related functions**: `pandas.api.types.is_scalar()` correctly identifies all these values as scalars
- **Documentation source**: The docstring is accessible via `pandas.api.types.infer_dtype.__doc__`
- **Workaround**: Users can wrap scalars in a list: `infer_dtype([value])` instead of `infer_dtype(value)`

The bug is particularly problematic because:
- Python's built-in numeric types are the most common scalar types users will encounter
- The selective support (works for str/numpy but not int/float) creates confusion about what's supported
- The function name suggests it should work on individual values, not just collections

## Proposed Fix

The function should detect Python built-in scalars and handle them appropriately before attempting iteration. Here's a minimal fix:

```diff
--- a/pandas/_libs/lib.pyx
+++ b/pandas/_libs/lib.pyx
@@ -1618,6 +1618,11 @@ def infer_dtype(object value, bint skipna=True) -> str:
         'mixed'
     """
+    # Handle Python built-in scalars by wrapping them in a list
+    if isinstance(value, (int, float, bool, complex, type(None))) and not isinstance(value, np.generic):
+        value = [value]
+    elif not hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
+        value = [value]
+
     cdef:
         Py_ssize_t i, n
         ndarray values
```

Alternatively, a more comprehensive fix would check for all scalar types using the existing `is_scalar` function from the same module:

```diff
--- a/pandas/_libs/lib.pyx
+++ b/pandas/_libs/lib.pyx
@@ -1618,6 +1618,10 @@ def infer_dtype(object value, bint skipna=True) -> str:
         'mixed'
     """
+    # Handle all scalar inputs consistently
+    from pandas.api.types import is_scalar
+    if is_scalar(value) and not isinstance(value, (str, bytes)):
+        value = [value]
+
     cdef:
         Py_ssize_t i, n
```