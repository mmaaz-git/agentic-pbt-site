# Bug Report: SparseArray Boolean Comparison Missing Implementation

**Target**: `pandas.core.arrays.sparse.array._sparse_array_op`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Comparing two boolean `SparseArray` objects using comparison operators (`==`, `!=`, `<`, `>`, etc.) crashes with `AttributeError` because the required Cython functions (e.g., `sparse_eq_bool`, `sparse_ne_bool`) are not implemented in `pandas._libs.sparse`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.arrays import SparseArray

@st.composite
def sparse_arrays(draw, min_size=0, max_size=100):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    values = draw(st.lists(st.booleans(), min_size=size, max_size=size))
    fill_value = draw(st.booleans())
    kind = draw(st.sampled_from(['integer', 'block']))
    return SparseArray(values, fill_value=fill_value, kind=kind)

@given(sparse_arrays(), sparse_arrays())
@settings(max_examples=100)
def test_equality_symmetric(arr1, arr2):
    """If a.equals(b), then b.equals(a)"""
    if arr1.equals(arr2):
        assert arr2.equals(arr1), "Equality not symmetric"
```

**Failing input**: Any two boolean SparseArrays where the optimized sparse comparison path is taken (different indices, same length, both have gaps).

## Reproducing the Bug

```python
from pandas.arrays import SparseArray

arr1 = SparseArray([False, False, True], fill_value=True)
arr2 = SparseArray([False, True, True], fill_value=True)

result = arr1 == arr2
```

**Output:**
```
AttributeError: module 'pandas._libs.sparse' has no attribute 'sparse_eq_bool'.
Did you mean: 'sparse_eq_int64'?
```

## Why This Is A Bug

The `_sparse_array_op` function attempts to dispatch comparison operations to optimized Cython functions in `pandas._libs.sparse`. For boolean operations, it constructs the function name as `sparse_{op}_{dtype}`, which results in names like:
- `sparse_eq_bool`
- `sparse_ne_bool`
- `sparse_lt_bool`
- etc.

However, only the following boolean operations are implemented:
- `sparse_and_uint8` (logical AND)
- `sparse_or_uint8` (logical OR)
- `sparse_xor_uint8` (logical XOR)

All comparison operators for boolean dtype are missing. The code has special handling for logical operations (`and`, `or`, `xor`) to use `uint8` functions, but no such handling exists for comparison operations.

Available functions in `pandas._libs.sparse`:
```
sparse_eq_float64    ✓
sparse_eq_int64      ✓
sparse_eq_bool       ✗  (missing!)
sparse_ne_float64    ✓
sparse_ne_int64      ✓
sparse_ne_bool       ✗  (missing!)
```

## Fix

Add fallback to dense operations for boolean comparisons, similar to how logical operations are handled:

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -232,7 +232,10 @@ def _sparse_array_op(
             left_sp_values = left.sp_values.view(np.uint8)
             right_sp_values = right.sp_values.view(np.uint8)
             result_dtype = bool
+        elif name in ("eq", "ne", "lt", "le", "gt", "ge") and dtype == "bool":
+            # Boolean comparison operations are not implemented in sparse lib
+            # Fall back to dense computation
+            with np.errstate(all="ignore"):
+                result = op(left.to_dense(), right.to_dense())
+                fill = op(_get_fill(left), _get_fill(right))
+            index = left.sp_index if left.sp_index.ngaps == 0 else right.sp_index
+            return _wrap_result(name, result, index, fill, dtype=result_dtype)
         else:
             opname = f"sparse_{name}_{dtype}"
             left_sp_values = left.sp_values
```

Alternatively, the boolean comparison operations could be implemented in the Cython layer, or they could reuse the `uint8` operations similar to logical operations.