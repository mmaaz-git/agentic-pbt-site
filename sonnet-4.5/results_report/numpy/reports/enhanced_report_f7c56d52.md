# Bug Report: numpy.ma Fill Value Functions Crash With Dtype Classes

**Target**: `numpy.ma.default_fill_value`, `numpy.ma.maximum_fill_value`, `numpy.ma.minimum_fill_value`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The numpy.ma fill value functions crash with AttributeError when passed numpy dtype classes (e.g., `np.int32`, `np.float64`) instead of dtype instances, breaking consistency with NumPy's convention of accepting both forms interchangeably throughout the API.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings

@given(st.sampled_from([np.int32, np.int64, np.float32, np.float64]))
@settings(max_examples=100)
def test_fill_value_functions_accept_dtype_classes(dtype):
    default = ma.default_fill_value(dtype)
    maximum = ma.maximum_fill_value(dtype)
    minimum = ma.minimum_fill_value(dtype)
    assert all(x is not None for x in [default, maximum, minimum])

# Run the test
test_fill_value_functions_accept_dtype_classes()
```

<details>

<summary>
**Failing input**: `numpy.int32`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 14, in <module>
    test_fill_value_functions_accept_dtype_classes()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 6, in test_fill_value_functions_accept_dtype_classes
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 8, in test_fill_value_functions_accept_dtype_classes
    default = ma.default_fill_value(dtype)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/core.py", line 313, in default_fill_value
    return _recursive_fill_value(dtype, _scalar_fill_value)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/core.py", line 233, in _recursive_fill_value
    if dtype.names is not None:
       ^^^^^^^^^^^
AttributeError: 'getset_descriptor' object has no attribute 'names'
Falsifying example: test_fill_value_functions_accept_dtype_classes(
    dtype=numpy.int32,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

# Test with dtype class (should fail)
print("Testing with dtype class np.int32:")
print("=" * 50)

for func_name, func in [('default_fill_value', ma.default_fill_value),
                         ('maximum_fill_value', ma.maximum_fill_value),
                         ('minimum_fill_value', ma.minimum_fill_value)]:
    try:
        fill = func(np.int32)
        print(f"{func_name}(np.int32): {fill}")
    except AttributeError as e:
        print(f"{func_name}(np.int32): FAILS - AttributeError: {e}")

print("\nTesting with dtype instance np.dtype('int32'):")
print("=" * 50)

for func_name, func in [('default_fill_value', ma.default_fill_value),
                         ('maximum_fill_value', ma.maximum_fill_value),
                         ('minimum_fill_value', ma.minimum_fill_value)]:
    try:
        fill = func(np.dtype('int32'))
        print(f"{func_name}(np.dtype('int32')): {fill}")
    except Exception as e:
        print(f"{func_name}(np.dtype('int32')): FAILS - {type(e).__name__}: {e}")
```

<details>

<summary>
AttributeError: All three functions fail with dtype classes
</summary>
```
Testing with dtype class np.int32:
==================================================
default_fill_value(np.int32): FAILS - AttributeError: 'getset_descriptor' object has no attribute 'names'
maximum_fill_value(np.int32): FAILS - AttributeError: 'getset_descriptor' object has no attribute 'names'
minimum_fill_value(np.int32): FAILS - AttributeError: 'getset_descriptor' object has no attribute 'names'

Testing with dtype instance np.dtype('int32'):
==================================================
default_fill_value(np.dtype('int32')): 999999
maximum_fill_value(np.dtype('int32')): -2147483648
minimum_fill_value(np.dtype('int32')): 2147483647
```
</details>

## Why This Is A Bug

This violates NumPy's established API convention where dtype parameters accept both dtype classes (`np.int32`) and dtype instances (`np.dtype('int32')`) interchangeably. Throughout NumPy, functions like `np.array()`, `np.zeros()`, and `np.ones()` accept both forms, creating a reasonable user expectation of consistency.

The bug occurs in the `_get_dtype_of()` helper function at numpy/ma/core.py:250-257. When given a dtype class like `np.int32`:
1. The `isinstance(obj, np.dtype)` check fails (dtype classes are not dtype instances)
2. The `hasattr(obj, 'dtype')` check succeeds (dtype classes have a dtype descriptor attribute)
3. It returns `obj.dtype`, which is a getset_descriptor object, not a dtype instance
4. This descriptor is passed to `_recursive_fill_value()` which attempts to access `.names`
5. The AttributeError occurs because getset_descriptor objects don't have a `.names` attribute

The documentation states these functions accept "dtype" parameters without specifying that only dtype instances are supported. Given NumPy's consistent pattern of accepting both dtype classes and instances elsewhere, this inconsistency breaks the principle of least surprise.

## Relevant Context

The numpy.ma module provides masked array functionality, and these fill value functions are commonly used to determine appropriate fill values for different data types when working with masked arrays. The functions work correctly with:
- Dtype instances: `np.dtype('int32')`, `np.dtype('float64')`
- Arrays: `np.array([1, 2, 3])`
- Scalar values: `42`, `3.14`

But fail specifically with dtype classes which are commonly used throughout NumPy code. This inconsistency can be particularly confusing for users who are accustomed to NumPy's usual flexibility with dtype specifications.

Related documentation: https://numpy.org/doc/stable/reference/maskedarray.generic.html#filling-values

## Proposed Fix

```diff
--- a/numpy/ma/core.py
+++ b/numpy/ma/core.py
@@ -250,6 +250,9 @@ def _get_dtype_of(obj):
     """ Convert the argument for *_fill_value into a dtype """
     if isinstance(obj, np.dtype):
         return obj
+    elif isinstance(obj, type) and issubclass(obj, np.generic):
+        # Handle dtype classes like np.int32, np.float64
+        return np.dtype(obj)
     elif hasattr(obj, 'dtype'):
         return obj.dtype
     else:
```