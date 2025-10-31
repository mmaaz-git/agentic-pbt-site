# Bug Report: xarray.core.duck_array_ops.sum_where Inverted Conditional Logic

**Target**: `xarray.core.duck_array_ops.sum_where`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `sum_where` function in xarray's duck_array_ops module implements inverted conditional logic - it sums array elements where the condition is `False` instead of `True`, contradicting both the function's name semantics and NumPy's established `where` parameter convention.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from xarray.core import duck_array_ops


@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=100),
    st.lists(st.booleans(), min_size=1, max_size=100)
)
@settings(max_examples=100)
def test_sum_where_matches_numpy(data_list, where_list):
    size = min(len(data_list), len(where_list))
    data = np.array(data_list[:size])
    where = np.array(where_list[:size])

    numpy_result = np.sum(data, where=where)
    xarray_result = duck_array_ops.sum_where(data, where=where)

    assert np.isclose(numpy_result, xarray_result), f"numpy: {numpy_result}, xarray: {xarray_result}, data: {data_list[:size]}, where: {where_list[:size]}"

if __name__ == "__main__":
    test_sum_where_matches_numpy()
```

<details>

<summary>
**Failing input**: `data_list=[1.0], where_list=[False]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 22, in <module>
    test_sum_where_matches_numpy()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 7, in test_sum_where_matches_numpy
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 19, in test_sum_where_matches_numpy
    assert np.isclose(numpy_result, xarray_result), f"numpy: {numpy_result}, xarray: {xarray_result}, data: {data_list[:size]}, where: {where_list[:size]}"
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: numpy: 0.0, xarray: 1.0, data: [1.0], where: [False]
Falsifying example: test_sum_where_matches_numpy(
    data_list=[1.0],
    where_list=[False],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from xarray.core import duck_array_ops

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
where = np.array([True, False, True, False, True])

numpy_result = np.sum(data, where=where)
xarray_result = duck_array_ops.sum_where(data, where=where)

print(f"Data array: {data}")
print(f"Where mask: {where}")
print(f"numpy.sum(data, where=where): {numpy_result}")
print(f"xarray sum_where(data, where=where): {xarray_result}")
print(f"Expected sum (1.0 + 3.0 + 5.0): 9.0")
print(f"Actual xarray sum (2.0 + 4.0): {xarray_result}")
print(f"\nAre results equal? {numpy_result == xarray_result}")

assert numpy_result != xarray_result, "Bug confirmed: sum_where has inverted logic"
print("\nBug confirmed: sum_where sums where condition is False instead of True")
```

<details>

<summary>
Output shows inverted logic: sums False positions instead of True
</summary>
```
Data array: [1. 2. 3. 4. 5.]
Where mask: [ True False  True False  True]
numpy.sum(data, where=where): 9.0
xarray sum_where(data, where=where): 6.0
Expected sum (1.0 + 3.0 + 5.0): 9.0
Actual xarray sum (2.0 + 4.0): 6.0

Are results equal? False

Bug confirmed: sum_where sums where condition is False instead of True
```
</details>

## Why This Is A Bug

The `sum_where` function violates the principle of least surprise and contradicts NumPy's established API convention. According to NumPy's documentation, the `where` parameter in `np.sum()` specifies which elements to **include** in the sum - when `where[i]` is `True`, element `i` is included; when `False`, it's excluded.

The bug occurs in the implementation at `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/core/duck_array_ops.py:389-396`. The problematic line is:

```python
a = where_method(xp.zeros_like(data), where, data)
```

This calls `where_method`, which internally calls `where(cond, data, other)`. The xarray `where` function follows NumPy semantics: it returns the first argument when the condition is `True`, and the second argument when `False`. So the current code:
- Returns `zeros_like(data)` (i.e., 0) when `where` is `True`
- Returns `data` (the actual values) when `where` is `False`

This is exactly backwards from the expected behavior. The function literally sums where the condition is `False` instead of `True`.

## Relevant Context

1. **Function location**: `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/core/duck_array_ops.py:389-396`

2. **Current usage**: The function is used by `nansum` in `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/computation/nanops.py:99`, which passes `mask = isnull(a)` as the `where` parameter. This works correctly only because the bug cancels out the inverted mask - `nansum` wants to sum non-null values, so it passes a mask of null values, and `sum_where` incorrectly sums where the mask is `False`, resulting in summing the non-null values. This "two wrongs make a right" pattern is fragile and confusing.

3. **Documentation status**: The `sum_where` function has no documentation or docstring. It appears to be an internal utility function not part of the public API.

4. **NumPy compatibility**: NumPy's official documentation states that `where` parameter elements with `True` values are included in the operation. Example: `np.sum([1, 2, 3], where=[True, False, True])` returns `4` (sum of 1 and 3).

5. **Impact scope**: While the bug doesn't currently break production code due to the compensating error in `nansum`, any direct use of `sum_where` expecting NumPy-like behavior will produce incorrect results. This creates a maintenance hazard and violates the principle of least surprise.

## Proposed Fix

```diff
--- a/xarray/core/duck_array_ops.py
+++ b/xarray/core/duck_array_ops.py
@@ -389,7 +389,7 @@ def sum_where(data, axis=None, dtype=None, where=None):
 def sum_where(data, axis=None, dtype=None, where=None):
     xp = get_array_namespace(data)
     if where is not None:
-        a = where_method(xp.zeros_like(data), where, data)
+        a = where_method(data, where, xp.zeros_like(data))
     else:
         a = data
     result = xp.sum(a, axis=axis, dtype=dtype)

--- a/xarray/computation/nanops.py
+++ b/xarray/computation/nanops.py
@@ -96,7 +96,7 @@ def nansum(a, axis=None, dtype=None, out=None, min_count=None):

 def nansum(a, axis=None, dtype=None, out=None, min_count=None):
     mask = isnull(a)
-    result = sum_where(a, axis=axis, dtype=dtype, where=mask)
+    result = sum_where(a, axis=axis, dtype=dtype, where=~mask)
     if min_count is not None:
         return _maybe_null_out(result, axis, mask, min_count)
     else:
```