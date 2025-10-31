# Bug Report: pandas SparseArray.density raises ZeroDivisionError on empty arrays

**Target**: `pandas.core.arrays.sparse.SparseArray.density`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `density` property of `SparseArray` raises an unhandled `ZeroDivisionError` when called on an empty array due to division by zero (`0 / 0`). This violates the expectation that property accessors should handle all valid object states gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import hypothesis.extra.numpy as npst
import numpy as np
from pandas.arrays import SparseArray

@given(
    data=npst.arrays(
        dtype=npst.integer_dtypes() | npst.floating_dtypes(),
        shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=0, max_side=100)
    )
)
@settings(max_examples=1000)
def test_density_property(data):
    sparse = SparseArray(data)

    if len(sparse) == 0:
        density = sparse.density

        assert not np.isnan(density), (
            f"BUG: density={density} for empty array (length=0). "
            f"Should return 0.0 or raise informative error."
        )
        assert not np.isinf(density), f"density should not be Inf for empty array"

    else:
        expected_density = sparse.npoints / len(sparse)
        assert sparse.density == expected_density

# Run the test
test_density_property()
```

<details>

<summary>
**Failing input**: `array([], dtype=int8)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 30, in <module>
    test_density_property()
    ~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 7, in test_density_property
    data=npst.arrays(
              ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 17, in test_density_property
    density = sparse.density
              ^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py", line 708, in density
    return self.sp_index.npoints / self.sp_index.length
           ~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: division by zero
Falsifying example: test_density_property(
    data=array([], dtype=int8),
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/11/hypo.py:17
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.arrays import SparseArray

# Create an empty sparse array
empty_sparse = SparseArray([])

# Print basic properties
print(f"len: {len(empty_sparse)}")
print(f"npoints: {empty_sparse.npoints}")

# Try to access the density property
try:
    density = empty_sparse.density
    print(f"density: {density}")
    print(f"Is NaN: {np.isnan(density)}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")
```

<details>

<summary>
Exception raised when accessing density on empty SparseArray
</summary>
```
len: 0
npoints: 0
Exception raised: ZeroDivisionError: division by zero
```
</details>

## Why This Is A Bug

This is a bug because:

1. **Unhandled exception on valid input**: Creating an empty SparseArray is valid (`SparseArray([])` succeeds), but accessing its `density` property crashes with an unhandled `ZeroDivisionError`.

2. **Violates property accessor contract**: Properties typed as returning `float` should not raise exceptions for valid object states. The docstring states it returns "The percent of non- ``fill_value`` points, as decimal" but makes no mention of potential exceptions.

3. **Mathematically undefined operation**: The implementation performs `self.sp_index.npoints / self.sp_index.length` which becomes `0 / 0` for empty arrays. This is mathematically undefined and should be explicitly handled.

4. **Inconsistent with array conventions**: Empty collections are common in data processing. Most array properties handle empty cases gracefully (e.g., `len()` returns 0, `npoints` returns 0).

5. **Poor user experience**: Users must defensively check array length before accessing a simple property, or risk unexpected crashes in production code.

## Relevant Context

- The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/sparse/array.py:708`
- The implementation directly divides without checking for zero length: `return self.sp_index.npoints / self.sp_index.length`
- Documentation: https://pandas.pydata.org/docs/reference/api/pandas.arrays.SparseArray.density.html
- No existing pandas tests cover the empty array case for the `density` property
- The issue has likely gone unnoticed due to lack of property-based testing in the pandas test suite
- pandas version tested: 2.3.2

## Proposed Fix

```diff
--- a/pandas/core/arrays/sparse/array.py
+++ b/pandas/core/arrays/sparse/array.py
@@ -696,7 +696,14 @@ class SparseArray(OpsMixin, PandasObject, ExtensionArray):
     @property
     def density(self) -> float:
         """
         The percent of non- ``fill_value`` points, as decimal.
+
+        For empty arrays, returns 0.0.

         Examples
         --------
         >>> from pandas.arrays import SparseArray
         >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
         >>> s.density
         0.6
+        >>> empty = SparseArray([])
+        >>> empty.density
+        0.0
         """
+        if self.sp_index.length == 0:
+            return 0.0
         return self.sp_index.npoints / self.sp_index.length
```