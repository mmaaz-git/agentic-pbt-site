# Bug Report: pandas.core.sparse.SparseArray.astype() Returns Wrong Type Instead of SparseArray

**Target**: `pandas.core.sparse.api.SparseArray.astype`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `astype()` method's docstring explicitly guarantees "The output will always be a SparseArray", but when converting to a non-SparseDtype (e.g., `np.float64`), it returns a numpy ndarray instead, violating the documented contract.

## Property-Based Test

```python
import numpy as np
from pandas.core.sparse.api import SparseArray
from hypothesis import given, strategies as st

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=100))
def test_astype_returns_sparsearray(data):
    sparse = SparseArray(data, dtype=np.int64)
    result = sparse.astype(np.float64)
    assert isinstance(result, SparseArray), \
        f"astype() should return SparseArray, got {type(result)}"

if __name__ == "__main__":
    test_astype_returns_sparsearray()
```

<details>

<summary>
**Failing input**: `[0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 13, in <module>
    test_astype_returns_sparsearray()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 6, in test_astype_returns_sparsearray
    def test_astype_returns_sparsearray(data):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 9, in test_astype_returns_sparsearray
    assert isinstance(result, SparseArray), \
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
AssertionError: astype() should return SparseArray, got <class 'numpy.ndarray'>
Falsifying example: test_astype_returns_sparsearray(
    data=[0],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.sparse.api import SparseArray

sparse = SparseArray([1, 2, 3], dtype=np.int64)
result = sparse.astype(np.float64)

print(f"Result type: {type(result)}")
print(f"Result value: {result}")
print(f"Is SparseArray: {isinstance(result, SparseArray)}")
```

<details>

<summary>
Output showing numpy.ndarray instead of SparseArray
</summary>
```
Result type: <class 'numpy.ndarray'>
Result value: [1. 2. 3.]
Is SparseArray: False
```
</details>

## Why This Is A Bug

The `astype()` method's docstring at lines 1241-1242 of `/pandas/core/arrays/sparse/array.py` explicitly states:

> "The output will always be a SparseArray. To convert to a dense ndarray with a certain dtype, use :meth:`numpy.asarray`."

This creates an unambiguous contract with users: `astype()` always returns a SparseArray, and if they want a dense numpy array, they should use `numpy.asarray()`. The docstring's return type is also documented as "SparseArray" (line 1258).

However, the actual implementation at lines 1301-1305 violates this contract. When the target dtype is not a `SparseDtype`, the method:
1. Converts the SparseArray to a dense numpy array using `np.asarray(self)`
2. Calls `astype_array()` which returns a numpy array
3. Returns this numpy array directly without wrapping it back in a SparseArray

This breaks user code that expects a SparseArray and tries to call SparseArray-specific methods on the result. For example, users cannot chain `.sp_values` or `.fill_value` after calling `astype()` with a non-sparse dtype.

## Relevant Context

The problematic code is in `/pandas/core/arrays/sparse/array.py` at lines 1301-1305. The comment "# GH#34457" suggests this behavior might have been introduced to fix GitHub issue #34457, but the documentation was never updated to reflect this change in behavior.

The documentation provides examples showing that `astype()` should preserve the sparse structure even when changing the dtype kind (e.g., from int to float). The method is designed to be distinct from `numpy.asarray()` which is the designated method for converting to dense arrays.

This bug affects a core data structure operation in pandas' sparse array implementation, potentially breaking code that relies on type preservation for memory efficiency in sparse data workflows.

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -1300,7 +1300,10 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
         future_dtype = pandas_dtype(dtype)
         if not isinstance(future_dtype, SparseDtype):
             # GH#34457
             values = np.asarray(self)
             values = ensure_wrapped_if_datetimelike(values)
-            return astype_array(values, dtype=future_dtype, copy=False)
+            converted = astype_array(values, dtype=future_dtype, copy=False)
+            # Wrap back in SparseArray to honor the docstring contract
+            return SparseArray(converted)

         dtype = self.dtype.update_dtype(dtype)
         subtype = pandas_dtype(dtype._subtype_with_str)
```