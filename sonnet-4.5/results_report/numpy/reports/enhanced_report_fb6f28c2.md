# Bug Report: numpy.ma.mask_or AttributeError with array_like inputs

**Target**: `numpy.ma.mask_or`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `numpy.ma.mask_or` function crashes with `AttributeError: 'NoneType' object has no attribute 'names'` when passed Python lists, despite its docstring explicitly stating it accepts "array_like" inputs.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume
import numpy.ma as ma
import numpy as np


@given(
    m1=st.lists(st.booleans(), min_size=1, max_size=50),
    m2=st.lists(st.booleans(), min_size=1, max_size=50)
)
def test_mask_or_symmetry(m1, m2):
    assume(len(m1) == len(m2))

    result1 = ma.mask_or(m1, m2)
    result2 = ma.mask_or(m2, m1)

    if result1 is ma.nomask and result2 is ma.nomask:
        pass
    elif result1 is ma.nomask or result2 is ma.nomask:
        assert False, f"mask_or should be symmetric, but one is nomask: {result1} vs {result2}"
    else:
        assert np.array_equal(result1, result2), f"mask_or not symmetric: {result1} vs {result2}"

if __name__ == "__main__":
    test_mask_or_symmetry()
```

<details>

<summary>
**Failing input**: `m1=[False], m2=[False]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 27, in <module>
    test_mask_or_symmetry()
    ~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 10, in test_mask_or_symmetry
    m1=st.lists(st.booleans(), min_size=1, max_size=50),
               ^^^
  File "/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 16, in test_mask_or_symmetry
    result1 = ma.mask_or(m1, m2)
  File "/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/ma/core.py", line 1808, in mask_or
    if dtype1.names is not None:
       ^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'names'
Falsifying example: test_mask_or_symmetry(
    m1=[False],  # or any other generated value
    m2=[False],
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages')

import numpy.ma as ma

m1 = [False]
m2 = [False]
result = ma.mask_or(m1, m2)
print(f"Result: {result}")
```

<details>

<summary>
AttributeError when calling ma.mask_or with Python lists
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/repo.py", line 8, in <module>
    result = ma.mask_or(m1, m2)
  File "/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/ma/core.py", line 1808, in mask_or
    if dtype1.names is not None:
       ^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'names'
```
</details>

## Why This Is A Bug

The `mask_or` function's documentation at line 1768 of `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/ma/core.py` explicitly states that its parameters `m1` and `m2` are "array_like". According to NumPy's official documentation, "array_like" includes any sequence that can be interpreted as an ndarray, including Python lists, tuples, and scalars.

The function crashes because at line 1805, it uses `getattr(m1, 'dtype', None)` to extract the dtype attribute. When Python lists are passed, this returns `None` since lists don't have a dtype attribute. The code then proceeds to line 1808 where it attempts to access `dtype1.names` without first checking if `dtype1` is `None`, causing the AttributeError.

The implementation already shows intent to handle non-array inputs through the use of `getattr` with default values at lines 1798 and 1805, but fails to complete the null check before accessing the `.names` attribute. This is a clear violation of the documented contract that the function should accept "array_like" inputs.

## Relevant Context

The function correctly handles cases where one input is `nomask` or `False` (lines 1797-1802), and the underlying `numpy.logical_or` function (called at line 1813) correctly handles Python lists. The bug only occurs in the specific code path at line 1808 that checks for structured arrays with named fields.

Other related functions like `ma.make_mask` (shown in the function's own examples) correctly handle Python lists, demonstrating that this is an inconsistency in the masked array module's API.

Function location: `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/ma/core.py:1760-1813`

## Proposed Fix

```diff
--- a/numpy/ma/core.py
+++ b/numpy/ma/core.py
@@ -1805,7 +1805,7 @@ def mask_or(m1, m2, copy=False, shrink=True):
     (dtype1, dtype2) = (getattr(m1, 'dtype', None), getattr(m2, 'dtype', None))
     if dtype1 != dtype2:
         raise ValueError(f"Incompatible dtypes '{dtype1}'<>'{dtype2}'")
-    if dtype1.names is not None:
+    if dtype1 is not None and dtype1.names is not None:
         # Allocate an output mask array with the properly broadcast shape.
         newmask = np.empty(np.broadcast(m1, m2).shape, dtype1)
         _recursive_mask_or(m1, m2, newmask)
```