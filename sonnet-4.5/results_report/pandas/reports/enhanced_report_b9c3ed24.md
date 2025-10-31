# Bug Report: pandas.core.sparse.SparseArray.astype Returns Wrong Type

**Target**: `pandas.core.arrays.sparse.array.SparseArray.astype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `astype()` method of `SparseArray` violates its documented contract by returning a numpy ndarray instead of a SparseArray when converting to non-SparseDtype types.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.arrays import SparseArray
import numpy as np

@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100))
def test_astype_preserves_values(data):
    sparse = SparseArray(data, dtype=np.int64)
    sparse_float = sparse.astype(np.float64)

    assert isinstance(sparse_float, SparseArray), f"Expected SparseArray, got {type(sparse_float)}"
    assert np.array_equal(sparse.to_dense(), sparse_float.to_dense())

if __name__ == "__main__":
    test_astype_preserves_values()
```

<details>

<summary>
**Failing input**: `[0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 14, in <module>
    test_astype_preserves_values()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 6, in test_astype_preserves_values
    def test_astype_preserves_values(data):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 10, in test_astype_preserves_values
    assert isinstance(sparse_float, SparseArray), f"Expected SparseArray, got {type(sparse_float)}"
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected SparseArray, got <class 'numpy.ndarray'>
Falsifying example: test_astype_preserves_values(
    data=[0],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from pandas.arrays import SparseArray
import numpy as np

sparse = SparseArray([1, 0, 0, 2], dtype=np.int64)
print(f"Original type: {type(sparse)}")
print(f"Original array: {sparse}")

sparse_float = sparse.astype(np.float64)
print(f"\nAfter astype(np.float64):")
print(f"Result type: {type(sparse_float)}")
print(f"Is SparseArray: {isinstance(sparse_float, SparseArray)}")
print(f"Result values: {sparse_float}")

# According to the documentation, astype() should always return a SparseArray
# But we're getting a numpy.ndarray instead
```

<details>

<summary>
Type mismatch: Expected SparseArray, got numpy.ndarray
</summary>
```
Original type: <class 'pandas.core.arrays.sparse.array.SparseArray'>
Original array: [1, 0, 0, 2]
Fill: 0
IntIndex
Indices: array([0, 3], dtype=int32)


After astype(np.float64):
Result type: <class 'numpy.ndarray'>
Is SparseArray: False
Result values: [1. 0. 0. 2.]
```
</details>

## Why This Is A Bug

The docstring for `SparseArray.astype()` at line 1241 of `pandas/core/arrays/sparse/array.py` explicitly states: **"The output will always be a SparseArray."** This is an unambiguous contract promise with no exceptions mentioned. The Returns section (lines 1256-1258) also clearly specifies "SparseArray" as the only return type.

The documentation further clarifies the distinction by stating: "To convert to a dense ndarray with a certain dtype, use :meth:`numpy.asarray`." This implies that `astype()` should NOT return a dense ndarray - that's what `numpy.asarray()` is for.

However, when converting to a non-SparseDtype (like `np.float64`), the implementation at lines 1301-1305 returns the result of `astype_array()` directly, which produces a numpy ndarray. This violates the documented API contract and breaks code that expects `astype()` to preserve the SparseArray type, such as method chaining or code that relies on sparse-specific methods.

## Relevant Context

The bug occurs in the code path at lines 1301-1305 of `/pandas/core/arrays/sparse/array.py`:

```python
if not isinstance(future_dtype, SparseDtype):
    # GH#34457
    values = np.asarray(self)
    values = ensure_wrapped_if_datetimelike(values)
    return astype_array(values, dtype=future_dtype, copy=False)  # Returns ndarray!
```

This code path is taken whenever converting to a non-SparseDtype (e.g., `np.float64` instead of `SparseDtype(np.float64)`). The comment "GH#34457" suggests this was an intentional change at some point, but it contradicts the documented behavior.

All examples in the docstring (lines 1262-1292) show `astype()` returning a SparseArray, even when converting to `float64`. The documentation is clear and consistent about the expected return type.

Pandas documentation link: https://pandas.pydata.org/docs/reference/api/pandas.arrays.SparseArray.astype.html
Source code: https://github.com/pandas-dev/pandas/blob/main/pandas/core/arrays/sparse/array.py#L1237

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1302,7 +1302,8 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
             # GH#34457
             values = np.asarray(self)
             values = ensure_wrapped_if_datetimelike(values)
-            return astype_array(values, dtype=future_dtype, copy=False)
+            result = astype_array(values, dtype=future_dtype, copy=False)
+            return type(self)(result, fill_value=self.fill_value)

         dtype = self.dtype.update_dtype(dtype)
         subtype = pandas_dtype(dtype._subtype_with_str)
```