# Bug Report: pandas.core.arrays.sparse argmin/argmax crash on all-fill-value arrays

**Target**: `pandas.core.arrays.sparse.SparseArray.argmin()` and `pandas.core.arrays.sparse.SparseArray.argmax()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

SparseArray's `argmin()` and `argmax()` methods crash with `ValueError: attempt to get argmin of an empty sequence` when the array contains only fill values, which occurs when all elements in the array equal the specified fill_value.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.arrays import SparseArray

@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    st.integers(min_value=-100, max_value=100),
)
@settings(max_examples=100)
def test_sparse_array_argmin_argmax_match_dense(data, fill_value):
    """
    Property: argmin() and argmax() should match dense array
    Evidence: _argmin_argmax method should find correct positions
    """
    sparse = SparseArray(data, fill_value=fill_value)
    dense = np.array(data)

    assert sparse.argmin() == dense.argmin()
    assert sparse.argmax() == dense.argmax()

if __name__ == "__main__":
    # Run the test
    test_sparse_array_argmin_argmax_match_dense()
```

<details>

<summary>
**Failing input**: `data=[0], fill_value=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 23, in <module>
    test_sparse_array_argmin_argmax_match_dense()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 6, in test_sparse_array_argmin_argmax_match_dense
    st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 18, in test_sparse_array_argmin_argmax_match_dense
    assert sparse.argmin() == dense.argmin()
           ~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1684, in argmin
    return self._argmin_argmax("argmin")
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1658, in _argmin_argmax
    _candidate = non_nan_idx[func(non_nans)]
                             ~~~~^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 1439, in argmin
    return _wrapfunc(a, 'argmin', axis=axis, out=out, **kwds)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 57, in _wrapfunc
    return bound(*args, **kwds)
ValueError: attempt to get argmin of an empty sequence
Falsifying example: test_sparse_array_argmin_argmax_match_dense(
    data=[0],
    fill_value=0,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:58
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.arrays import SparseArray

data = [0]
fill_value = 0

sparse = SparseArray(data, fill_value=fill_value)
dense = np.array(data)

print("Dense array argmin result:", dense.argmin())
print("Dense array argmax result:", dense.argmax())

print("\nAttempting sparse array argmin...")
try:
    result = sparse.argmin()
    print("Sparse array argmin result:", result)
except Exception as e:
    print(f"Error in sparse.argmin(): {type(e).__name__}: {e}")

print("\nAttempting sparse array argmax...")
try:
    result = sparse.argmax()
    print("Sparse array argmax result:", result)
except Exception as e:
    print(f"Error in sparse.argmax(): {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError: attempt to get argmin/argmax of an empty sequence
</summary>
```
Dense array argmin result: 0
Dense array argmax result: 0

Attempting sparse array argmin...
Error in sparse.argmin(): ValueError: attempt to get argmin of an empty sequence

Attempting sparse array argmax...
Error in sparse.argmax(): ValueError: attempt to get argmax of an empty sequence
```
</details>

## Why This Is A Bug

This is a bug because it violates the expected behavior of SparseArray methods in multiple ways:

1. **Valid input causes crash**: A SparseArray where all values equal the fill_value is a perfectly valid sparse array. In fact, it represents the most efficient sparse representation possible (zero stored values). The constructor accepts this configuration without error, so the methods should handle it properly.

2. **Inconsistent with NumPy behavior**: NumPy's `argmin()` and `argmax()` correctly return 0 for arrays where all values are equal (e.g., `np.array([0]).argmin()` returns 0). SparseArray should maintain compatibility with NumPy's behavior for equivalent logical arrays.

3. **Unhelpful error message**: The error "attempt to get argmin of an empty sequence" doesn't indicate the actual problem to users. The array itself is not empty (it has elements), only the internal `sp_values` array is empty because all values match the fill_value.

4. **Code already has logic to handle this case**: The `_argmin_argmax` method already contains logic to handle fill values via the `_first_fill_value_loc()` method (lines 1667-1672), but the crash occurs before this logic can be reached because `np.argmin()` is called on an empty `non_nans` array at line 1658.

## Relevant Context

The bug occurs specifically in the `_argmin_argmax` method at line 1658 of `/pandas/core/arrays/sparse/array.py`. The issue arises when:

1. A SparseArray is created with all values equal to the fill_value
2. This results in `self._sparse_values` being an empty array (no values need to be explicitly stored)
3. After filtering NaN values, `non_nans` becomes an empty array
4. Calling `func(non_nans)` where func is `np.argmin` or `np.argmax` raises ValueError

The sparse array format is working correctly by not storing any explicit values when everything equals the fill_value. The bug is that the argmin/argmax implementation doesn't handle this valid edge case.

Documentation reference: The pandas documentation states that SparseArray provides "an ExtensionArray for storing sparse data" and that argmin/argmax should "return the index of minimum/maximum value." There's no documented exception for arrays with all fill values.

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1655,6 +1655,11 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         non_nans = values[~mask]
         non_nan_idx = idx[~mask]

+        # Handle case where all values are fill_value (no sparse values)
+        if len(non_nans) == 0:
+            _loc = self._first_fill_value_loc()
+            return _loc if _loc != -1 else 0
+
         _candidate = non_nan_idx[func(non_nans)]
         candidate = index[_candidate]
```