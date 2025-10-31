# Bug Report: xarray.core.utils.is_uniform_spaced ValueError on Small Arrays

**Target**: `xarray.core.utils.is_uniform_spaced`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_uniform_spaced` function crashes with a ValueError when passed arrays with fewer than 2 elements due to calling `.min()` and `.max()` on empty arrays returned by `np.diff()`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.core.utils import is_uniform_spaced
import numpy as np

@given(n=st.integers(min_value=0, max_value=100))
@settings(max_examples=300)
def test_linspace_always_uniform(n):
    arr = np.linspace(0, 10, n)
    result = is_uniform_spaced(arr)
    assert result == True, f"linspace with {n} points should be uniformly spaced"

@given(size=st.integers(min_value=0, max_value=2))
@settings(max_examples=100)
def test_small_arrays_dont_crash(size):
    arr = list(range(size))
    result = is_uniform_spaced(arr)
    assert isinstance(result, bool)
```

<details>

<summary>
**Failing input**: `n=0` (empty array) and `size=0` (empty list)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 25, in <module>
    test_linspace_always_uniform()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 6, in test_linspace_always_uniform
    @settings(max_examples=300)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 9, in test_linspace_always_uniform
    result = is_uniform_spaced(arr)
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/core/utils.py", line 737, in is_uniform_spaced
    return bool(np.isclose(diffs.min(), diffs.max(), **kwargs))
                           ~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_methods.py", line 47, in _amin
    return umr_minimum(a, axis, None, out, keepdims, initial, where)
ValueError: zero-size array to reduction operation minimum which has no identity
Falsifying example: test_linspace_always_uniform(
    n=0,
)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 38, in <module>
    test_small_arrays_dont_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 13, in test_small_arrays_dont_crash
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 16, in test_small_arrays_dont_crash
    result = is_uniform_spaced(arr)
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/core/utils.py", line 737, in is_uniform_spaced
    return bool(np.isclose(diffs.min(), diffs.max(), **kwargs))
                           ~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_methods.py", line 47, in _amin
    return umr_minimum(a, axis, None, out, keepdims, initial, where)
ValueError: zero-size array to reduction operation minimum which has no identity
Falsifying example: test_small_arrays_dont_crash(
    size=0,
)
Running test_linspace_always_uniform...
test_linspace_always_uniform: FAILED
First failure with n=0 (empty array)
Error: ValueError: zero-size array to reduction operation minimum which has no identity

Traceback:

==================================================

Running test_small_arrays_dont_crash...
test_small_arrays_dont_crash: FAILED
First failure with size=0 (empty array)
Error: ValueError: zero-size array to reduction operation minimum which has no identity

Traceback:
```
</details>

## Reproducing the Bug

```python
from xarray.core.utils import is_uniform_spaced

# Test empty array
try:
    result = is_uniform_spaced([])
    print(f"Empty array result: {result}")
except Exception as e:
    print(f"Empty array error: {type(e).__name__}: {e}")

# Test single element array
try:
    result = is_uniform_spaced([5])
    print(f"Single element array result: {result}")
except Exception as e:
    print(f"Single element array error: {type(e).__name__}: {e}")

# Test two element array (should work)
try:
    result = is_uniform_spaced([1, 2])
    print(f"Two element array result: {result}")
except Exception as e:
    print(f"Two element array error: {type(e).__name__}: {e}")

# Test normal array (should work)
try:
    result = is_uniform_spaced([1, 2, 3, 4, 5])
    print(f"Normal array result: {result}")
except Exception as e:
    print(f"Normal array error: {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError: zero-size array to reduction operation minimum which has no identity
</summary>
```
Empty array error: ValueError: zero-size array to reduction operation minimum which has no identity
Single element array error: ValueError: zero-size array to reduction operation minimum which has no identity
Two element array result: True
Normal array result: True
```
</details>

## Why This Is A Bug

This violates expected behavior because:

1. **Undocumented restriction**: The function's docstring states it returns "True if values of an array are uniformly spaced and sorted" but doesn't mention any minimum array size requirement. The function signature accepts any array-like input.

2. **Crash instead of logical return**: The function crashes with an internal numpy ValueError instead of returning a boolean. For empty arrays and single-element arrays, the mathematical question "is this uniformly spaced?" has a clear answer - vacuously true, as there are no pairs of adjacent elements that could have different spacing.

3. **Inconsistent with numpy ecosystem**: Functions like `np.linspace(0, 10, 0)` and `np.linspace(0, 10, 1)` are valid calls that produce arrays of length 0 and 1 respectively. Passing these valid numpy outputs to `is_uniform_spaced` causes a crash.

4. **Internal implementation detail leaking**: The crash occurs at line 737 in utils.py when `np.diff(arr)` returns an empty array for inputs with <2 elements, and calling `.min()` or `.max()` on empty arrays raises ValueError. This is an implementation detail that shouldn't be exposed to callers.

## Relevant Context

The `is_uniform_spaced` function is located in `/home/npc/miniconda/lib/python3.13/site-packages/xarray/core/utils.py` at lines 725-737. It's an internal utility function (not part of the public API) primarily used by xarray's plotting system to determine if coordinates are evenly spaced.

The crash occurs because:
- When input has 0 elements: `np.diff([])` returns an empty array
- When input has 1 element: `np.diff([x])` returns an empty array
- Calling `.min()` or `.max()` on an empty numpy array raises ValueError

The function works correctly for arrays with 2+ elements because `np.diff()` returns a non-empty array.

Source code reference: [xarray/core/utils.py:725-737](https://github.com/pydata/xarray/blob/main/xarray/core/utils.py#L725-L737)

## Proposed Fix

```diff
def is_uniform_spaced(arr, **kwargs) -> bool:
    """Return True if values of an array are uniformly spaced and sorted.

    >>> is_uniform_spaced(range(5))
    True
    >>> is_uniform_spaced([-4, 0, 100])
    False

    kwargs are additional arguments to ``np.isclose``
    """
    arr = np.array(arr, dtype=float)
+   if len(arr) < 2:
+       return True
    diffs = np.diff(arr)
    return bool(np.isclose(diffs.min(), diffs.max(), **kwargs))
```