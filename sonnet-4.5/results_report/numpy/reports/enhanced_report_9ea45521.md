# Bug Report: numpy.ma.default_fill_value Crashes on NumPy Type Objects

**Target**: `numpy.ma.default_fill_value`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`numpy.ma.default_fill_value` crashes with an AttributeError when passed NumPy type objects (e.g., `np.float32`, `np.int64`) instead of dtype instances, due to incorrect assumptions about objects with a `dtype` attribute.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st

@given(st.data())
def test_default_fill_value_matches_type(data_strategy):
    dtype = data_strategy.draw(st.sampled_from([np.float32, np.float64, np.int32, np.int64]))

    fill_val = ma.default_fill_value(dtype)

    assert np.isscalar(fill_val) or fill_val is not None

# Run the test
test_default_fill_value_matches_type()
```

<details>

<summary>
**Failing input**: `dtype=numpy.float32`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 14, in <module>
    test_default_fill_value_matches_type()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 6, in test_default_fill_value_matches_type
    def test_default_fill_value_matches_type(data_strategy):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 9, in test_default_fill_value_matches_type
    fill_val = ma.default_fill_value(dtype)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/core.py", line 313, in default_fill_value
    return _recursive_fill_value(dtype, _scalar_fill_value)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/core.py", line 233, in _recursive_fill_value
    if dtype.names is not None:
       ^^^^^^^^^^^
AttributeError: 'getset_descriptor' object has no attribute 'names'
Falsifying example: test_default_fill_value_matches_type(
    data_strategy=data(...),
)
Draw 1: numpy.float32
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

# This should work but crashes with AttributeError
result = ma.default_fill_value(np.float32)
print(f"Result: {result}")
```

<details>

<summary>
AttributeError: 'getset_descriptor' object has no attribute 'names'
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/24/repo.py", line 5, in <module>
    result = ma.default_fill_value(np.float32)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/core.py", line 313, in default_fill_value
    return _recursive_fill_value(dtype, _scalar_fill_value)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/ma/core.py", line 233, in _recursive_fill_value
    if dtype.names is not None:
       ^^^^^^^^^^^
AttributeError: 'getset_descriptor' object has no attribute 'names'
```
</details>

## Why This Is A Bug

This violates expected behavior because the function crashes with an obscure internal error instead of either:
1. Correctly handling the input by converting `np.float32` to `np.dtype(np.float32)`
2. Raising a clear, informative error about incorrect input type

The crash occurs in the internal helper function `_get_dtype_of` (at line 254-255 in numpy/ma/core.py) which incorrectly assumes that if an object has a `dtype` attribute, accessing `obj.dtype` will return a valid NumPy dtype instance. However, NumPy type objects like `np.float32` have a `dtype` attribute that is a getset_descriptor, not a dtype instance. This descriptor is then passed to `_recursive_fill_value` which tries to access `dtype.names`, causing the AttributeError.

The documentation states the function accepts "ndarray, dtype or scalar" as input. While `np.float32` is technically a type class rather than a dtype instance, users commonly confuse these two forms throughout NumPy's ecosystem, and many NumPy functions accept both interchangeably (e.g., `np.array([1,2,3], dtype=np.float32)` works perfectly).

## Relevant Context

- NumPy type objects (`np.float32`, `np.int64`, etc.) are type classes that represent scalar types
- NumPy dtype instances (`np.dtype(np.float32)`) are the actual dtype objects expected by this function
- The distinction is subtle: `np.float32` is `<class 'numpy.float32'>` while `np.dtype(np.float32)` is a dtype instance
- The function already tries to be flexible with input types through the `_get_dtype_of` helper function
- The correct behavior can be seen by using: `ma.default_fill_value(np.dtype(np.float32))` which returns `1e+20`
- The examples in the documentation specifically show `np.dtype(complex)` rather than just `complex` or `np.complex128`, hinting at the correct usage

Documentation reference: https://numpy.org/doc/stable/reference/generated/numpy.ma.default_fill_value.html
Source code location: numpy/ma/core.py lines 250-313

## Proposed Fix

```diff
--- a/numpy/ma/core.py
+++ b/numpy/ma/core.py
@@ -250,7 +250,11 @@ def _recursive_fill_value(dtype, f):
 def _get_dtype_of(obj):
     """ Convert the argument for *_fill_value into a dtype """
     if isinstance(obj, np.dtype):
         return obj
+    elif isinstance(obj, type) and issubclass(obj, np.generic):
+        # Handle numpy type objects like np.float32, np.int64
+        return np.dtype(obj)
     elif hasattr(obj, 'dtype'):
         return obj.dtype
     else:
         return np.asanyarray(obj).dtype
```