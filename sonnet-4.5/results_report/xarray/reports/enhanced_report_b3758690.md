# Bug Report: xarray.namedarray.NamedArray.permute_dims() Returns Untransposed Data with Duplicate Dimension Names

**Target**: `xarray.namedarray.core.NamedArray.permute_dims`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`NamedArray.permute_dims()` fails to transpose array data when dimension names are duplicates, returning an unchanged copy instead of the transposed array. This violates the method's documented behavior of reversing dimension order by default.

## Property-Based Test

```python
import numpy as np
import warnings
from hypothesis import given, strategies as st, settings
from xarray.namedarray.core import NamedArray


@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=50)
def test_permute_dims_with_duplicate_names_transposes_data(rows, cols):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        arr = NamedArray(("x", "x"), np.arange(rows * cols).reshape(rows, cols))

    result = arr.permute_dims()

    np.testing.assert_array_equal(result.to_numpy(), arr.to_numpy().T,
                                   err_msg="permute_dims() should transpose data even with duplicate dimension names")

if __name__ == "__main__":
    test_permute_dims_with_duplicate_names_transposes_data()
```

<details>

<summary>
**Failing input**: `rows=2, cols=2`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/xarray/namedarray/core.py:261: UserWarning: Duplicate dimension names present: dimensions {'x'} appear more than once in dims=('x', 'x'). We do not yet support duplicate dimension names, but we do allow initial construction of the object. We recommend you rename the dims immediately to become distinct, as most xarray functionality is likely to fail silently if you do not. To rename the dimensions you will need to set the ``.dims`` attribute of each variable, ``e.g. var.dims=('x0', 'x1')``.
  self._dims = self._parse_dimensions(dims)
/home/npc/miniconda/lib/python3.13/site-packages/xarray/namedarray/core.py:261: UserWarning: Duplicate dimension names present: dimensions {'x'} appear more than once in dims=('x', 'x'). We do not yet support duplicate dimension names, but we do allow initial construction of the object. We recommend you rename the dims immediately to become distinct, as most xarray functionality is likely to fail silently if you do not. To rename the dimensions you will need to set the ``.dims`` attribute of each variable, ``e.g. var.dims=('x0', 'x1')``.
  self._dims = self._parse_dimensions(dims)
/home/npc/miniconda/lib/python3.13/site-packages/xarray/namedarray/core.py:261: UserWarning: Duplicate dimension names present: dimensions {'x'} appear more than once in dims=('x', 'x'). We do not yet support duplicate dimension names, but we do allow initial construction of the object. We recommend you rename the dims immediately to become distinct, as most xarray functionality is likely to fail silently if you do not. To rename the dimensions you will need to set the ``.dims`` attribute of each variable, ``e.g. var.dims=('x0', 'x1')``.
  self._dims = self._parse_dimensions(dims)
/home/npc/miniconda/lib/python3.13/site-packages/xarray/namedarray/core.py:261: UserWarning: Duplicate dimension names present: dimensions {'x'} appear more than once in dims=('x', 'x'). We do not yet support duplicate dimension names, but we do allow initial construction of the object. We recommend you rename the dims immediately to become distinct, as most xarray functionality is likely to fail silently if you do not. To rename the dimensions you will need to set the ``.dims`` attribute of each variable, ``e.g. var.dims=('x0', 'x1')``.
  self._dims = self._parse_dimensions(dims)
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 23, in <module>
  |     test_permute_dims_with_duplicate_names_transposes_data()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 8, in test_permute_dims_with_duplicate_names_transposes_data
  |     st.integers(min_value=2, max_value=5),
  |                ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 19, in test_permute_dims_with_duplicate_names_transposes_data
    |     np.testing.assert_array_equal(result.to_numpy(), arr.to_numpy().T,
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                                    err_msg="permute_dims() should transpose data even with duplicate dimension names")
    |                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 1051, in assert_array_equal
    |     assert_array_compare(operator.__eq__, actual, desired, err_msg=err_msg,
    |     ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                          verbose=verbose, header='Arrays are not equal',
    |                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                          strict=strict)
    |                          ^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 803, in assert_array_compare
    |     raise AssertionError(msg)
    | AssertionError:
    | Arrays are not equal
    | permute_dims() should transpose data even with duplicate dimension names
    | (shapes (2, 3), (3, 2) mismatch)
    |  ACTUAL: array([[0, 1, 2],
    |        [3, 4, 5]])
    |  DESIRED: array([[0, 3],
    |        [1, 4],
    |        [2, 5]])
    | Falsifying example: test_permute_dims_with_duplicate_names_transposes_data(
    |     rows=2,
    |     cols=3,
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 19, in test_permute_dims_with_duplicate_names_transposes_data
    |     np.testing.assert_array_equal(result.to_numpy(), arr.to_numpy().T,
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                                    err_msg="permute_dims() should transpose data even with duplicate dimension names")
    |                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 1051, in assert_array_equal
    |     assert_array_compare(operator.__eq__, actual, desired, err_msg=err_msg,
    |     ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                          verbose=verbose, header='Arrays are not equal',
    |                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                          strict=strict)
    |                          ^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/testing/_private/utils.py", line 916, in assert_array_compare
    |     raise AssertionError(msg)
    | AssertionError:
    | Arrays are not equal
    | permute_dims() should transpose data even with duplicate dimension names
    | Mismatched elements: 2 / 4 (50%)
    | Max absolute difference among violations: 1
    | Max relative difference among violations: 1.
    |  ACTUAL: array([[0, 1],
    |        [2, 3]])
    |  DESIRED: array([[0, 2],
    |        [1, 3]])
    | Falsifying example: test_permute_dims_with_duplicate_names_transposes_data(
    |     rows=2,
    |     cols=2,
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import numpy as np
import warnings

# Suppress the duplicate dimension warning globally for this test
warnings.filterwarnings("ignore", "Duplicate dimension names present")

from xarray.namedarray.core import NamedArray

# Create a NamedArray with duplicate dimension names
# This is intentional to demonstrate the bug
arr = NamedArray(("x", "x"), np.array([[1, 2], [3, 4]]))

print("Original:")
print(arr.to_numpy())

result = arr.permute_dims()

print("\nAfter permute_dims():")
print(result.to_numpy())

print("\nExpected (transposed):")
print(arr.to_numpy().T)
```

<details>

<summary>
Output shows permute_dims() fails to transpose with duplicate dimension names
</summary>
```
Original:
[[1 2]
 [3 4]]

After permute_dims():
[[1 2]
 [3 4]]

Expected (transposed):
[[1 3]
 [2 4]]
```
</details>

## Why This Is A Bug

The `permute_dims()` method documentation explicitly states: "By default, reverse the order of the dimensions." For a 2D array, this means transposing the data, swapping rows and columns. The method references `numpy.transpose`, which always transposes the underlying data regardless of any metadata about dimension names.

However, when dimension names are duplicates (e.g., `("x", "x")`), the method incorrectly returns an unchanged copy of the array instead of transposing it. This happens because of a logic error in the implementation at `/home/npc/miniconda/lib/python3.13/site-packages/xarray/namedarray/core.py:1043`:

```python
if len(dims) < 2 or dims == self.dims:
    return self.copy(deep=False)
```

When dimension names are duplicates like `("x", "x")`, reversing them produces `("x", "x")` which equals `self.dims`. The code mistakenly concludes that no transposition is needed since the dimension names are the same, but it should actually check whether the underlying axis order has changed, not just the names.

This is precisely the type of "silent failure" mentioned in the duplicate dimension warning that appears when creating such arrays: "most xarray functionality is likely to fail silently if you do not [rename dimensions]." The data is returned unchanged when it should be transposed, potentially causing incorrect calculations in downstream code.

## Relevant Context

1. **Duplicate dimension names are explicitly allowed**: While xarray warns about duplicate dimension names and states they are "not yet supported," it explicitly allows creating such objects. The warning states: "we do allow initial construction of the object."

2. **The warning acknowledges potential silent failures**: The warning itself mentions that "most xarray functionality is likely to fail silently" with duplicate dimension names. This bug is an example of such a silent failure.

3. **Documentation link**: The `permute_dims` method is documented at line 1033 in the source: `/home/npc/miniconda/lib/python3.13/site-packages/xarray/namedarray/core.py`

4. **NumPy compatibility expectation**: The method references `numpy.transpose` in its "See Also" section, suggesting it should behave similarly. NumPy's transpose always swaps axes regardless of any metadata.

5. **The T property relies on permute_dims()**: Line 1061 shows that the `.T` property for 2D arrays calls `permute_dims()`, so this bug also affects the transpose property.

## Proposed Fix

```diff
--- a/xarray/namedarray/core.py
+++ b/xarray/namedarray/core.py
@@ -1040,15 +1040,15 @@ class NamedArray(NamedArrayAggregations, Generic[_ShapeType_co, _DType_co]):
         else:
             dims = tuple(infix_dims(dim, self.dims, missing_dims))

-        if len(dims) < 2 or dims == self.dims:
+        axes = self.get_axis_num(dims)
+        assert isinstance(axes, tuple)
+
+        if len(dims) < 2 or axes == tuple(range(self.ndim)):
             # no need to transpose if only one dimension
-            # or dims are in same order
+            # or axes are in same order
             return self.copy(deep=False)

-        axes = self.get_axis_num(dims)
-        assert isinstance(axes, tuple)
-
         return permute_dims(self, axes)
```