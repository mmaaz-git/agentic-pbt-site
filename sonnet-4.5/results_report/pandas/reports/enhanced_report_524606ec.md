# Bug Report: pandas SparseArray Boolean Comparison Operations Crash

**Target**: `pandas.core.arrays.sparse.array._sparse_array_op`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Comparing two boolean SparseArray objects using any comparison operator (==, !=, <, >, <=, >=) causes an AttributeError crash because the required Cython functions for boolean comparisons are not implemented in pandas._libs.sparse.

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

<details>

<summary>
**Failing input**: `arr1=[False, False, False, False, False, False, False, False, False, False, False, False, False, False, True], arr2=[False, False, False, False, False, False, False, False, False, False, False, False, False, True, False]`
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
/home/npc/pbt/agentic-pbt/worker_/39/hypo.py:10: FutureWarning: Allowing arbitrary scalar fill_value in SparseDtype is deprecated. In a future version, the fill_value must be a valid value for the SparseDtype.subtype.
  return SparseArray(values, fill_value=fill_value, kind=kind)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 20, in <module>
    test_equality_symmetric()
    ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 13, in test_equality_symmetric
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 16, in test_equality_symmetric
    if arr1.equals(arr2):
       ~~~~~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/base.py", line 1367, in equals
    equal_values = self == other
                   ^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/ops/common.py", line 76, in new_method
    return method(self, other)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arraylike.py", line 40, in __eq__
    return self._cmp_method(other, operator.eq)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1811, in _cmp_method
    return _sparse_array_op(self, other, op, op_name)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 237, in _sparse_array_op
    sparse_op = getattr(splib, opname)
AttributeError: module 'pandas._libs.sparse' has no attribute 'sparse_eq_bool'. Did you mean: 'sparse_eq_int64'?
Falsifying example: test_equality_symmetric(
    arr1=[False, False, False, False, False, False, False, False, False, False, False, False, False, False, True]
    Fill: True
    IntIndex
    Indices: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13],
          dtype=int32),
    arr2=[False, False, False, False, False, False, False, False, False, False, False, False, False, True, False]
    Fill: True
    IntIndex
    Indices: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 14],
          dtype=int32),
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py:223
```
</details>

## Reproducing the Bug

```python
from pandas.arrays import SparseArray

# Create two boolean SparseArrays
arr1 = SparseArray([False, False, True], fill_value=True)
arr2 = SparseArray([False, True, True], fill_value=True)

# Attempt to compare them using equality operator
result = arr1 == arr2
```

<details>

<summary>
AttributeError: module 'pandas._libs.sparse' has no attribute 'sparse_eq_bool'
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/repo.py", line 8, in <module>
    result = arr1 == arr2
             ^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/ops/common.py", line 76, in new_method
    return method(self, other)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arraylike.py", line 40, in __eq__
    return self._cmp_method(other, operator.eq)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 1811, in _cmp_method
    return _sparse_array_op(self, other, op, op_name)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 237, in _sparse_array_op
    sparse_op = getattr(splib, opname)
AttributeError: module 'pandas._libs.sparse' has no attribute 'sparse_eq_bool'. Did you mean: 'sparse_eq_int64'?
```
</details>

## Why This Is A Bug

This is a bug because comparison operations are fundamental operations that should work for all data types, including boolean SparseArrays. The code crashes ungracefully with an AttributeError when attempting basic comparison operations, violating several expected behaviors:

1. **Incomplete Implementation**: The `_sparse_array_op` function in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py:223` constructs function names like `sparse_{op}_{dtype}` expecting them to exist in `pandas._libs.sparse`, but the boolean comparison functions are completely missing.

2. **Inconsistent Behavior**: Logical operations (and, or, xor) work correctly for boolean SparseArrays because they have special handling to use uint8 variants (lines 216-221), but comparison operations lack this special handling and crash instead.

3. **Ungraceful Failure**: Rather than falling back to dense operations or providing a helpful error message, the code crashes with an AttributeError that exposes internal implementation details.

4. **Violates Interface Contract**: SparseArray inherits from ExtensionArray and should support standard comparison operations. The documentation does not indicate that boolean dtypes should have limited functionality.

## Relevant Context

Testing confirms that ALL comparison operators fail for boolean SparseArrays:
- `==` (eq) → Missing `sparse_eq_bool`
- `!=` (ne) → Missing `sparse_ne_bool`
- `<` (lt) → Missing `sparse_lt_bool`
- `<=` (le) → Missing `sparse_le_bool`
- `>` (gt) → Missing `sparse_gt_bool`
- `>=` (ge) → Missing `sparse_ge_bool`

Meanwhile, logical operators work correctly:
- `&` (and) → Uses `sparse_and_uint8`
- `|` (or) → Uses `sparse_or_uint8`
- `^` (xor) → Uses `sparse_xor_uint8`

The available functions in `pandas._libs.sparse` include comparison operations for float64 and int64 dtypes, but not for bool dtype. This appears to be an implementation gap rather than a design decision.

Source code location: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py:216-237`

## Proposed Fix

Add special handling for boolean comparison operations similar to how logical operations are handled, or fall back to dense operations when sparse functions are unavailable:

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -219,6 +219,15 @@ def _sparse_array_op(
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
+            return _wrap_result(name, result[index.indices], index, fill, dtype=bool)
         else:
             opname = f"sparse_{name}_{dtype}"
             left_sp_values = left.sp_values
```