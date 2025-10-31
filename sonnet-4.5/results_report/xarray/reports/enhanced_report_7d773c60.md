# Bug Report: xarray.namedarray.core.NamedArray.permute_dims Ignores missing_dims Parameter

**Target**: `xarray.namedarray.core.NamedArray.permute_dims`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `permute_dims` method raises a ValueError for missing dimensions even when `missing_dims` is set to 'ignore' or 'warn', violating its documented behavior of gracefully handling missing dimensions.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import numpy as np
from xarray.namedarray.core import NamedArray


@given(st.integers(min_value=1, max_value=10))
@settings(max_examples=100)
def test_permute_dims_with_missing_dim_ignore(n):
    data = np.arange(n * 2).reshape(n, 2)
    arr = NamedArray(('x', 'y'), data)

    result = arr.permute_dims('x', 'z', missing_dims='ignore')

    assert result.dims == ('x', 'y')


if __name__ == "__main__":
    # Run the test
    test_permute_dims_with_missing_dim_ignore()
```

<details>

<summary>
**Failing input**: `n=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 19, in <module>
    test_permute_dims_with_missing_dim_ignore()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 7, in test_permute_dims_with_missing_dim_ignore
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 12, in test_permute_dims_with_missing_dim_ignore
    result = arr.permute_dims('x', 'z', missing_dims='ignore')
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/namedarray/core.py", line 1041, in permute_dims
    dims = tuple(infix_dims(dim, self.dims, missing_dims))  # type: ignore[arg-type]
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/namedarray/utils.py", line 173, in infix_dims
    raise ValueError(
        f"{dims_supplied} must be a permuted list of {dims_all}, unless `...` is included"
    )
ValueError: ('x', 'z') must be a permuted list of ('x', 'y'), unless `...` is included
Falsifying example: test_permute_dims_with_missing_dim_ignore(
    n=1,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from xarray.namedarray.core import NamedArray

# Create a simple NamedArray with 2 dimensions
data = np.arange(6).reshape(2, 3)
arr = NamedArray(('x', 'y'), data)

print("Original array dimensions:", arr.dims)
print("Original array shape:", arr.shape)
print()

# Try to permute dims with a missing dimension 'z' using missing_dims='ignore'
print("Attempting: arr.permute_dims('x', 'z', missing_dims='ignore')")
try:
    result = arr.permute_dims('x', 'z', missing_dims='ignore')
    print("Result dimensions:", result.dims)
    print("Result shape:", result.shape)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError raised despite missing_dims='ignore'
</summary>
```
Original array dimensions: ('x', 'y')
Original array shape: (2, 3)

Attempting: arr.permute_dims('x', 'z', missing_dims='ignore')
ERROR: ValueError: ('x', 'z') must be a permuted list of ('x', 'y'), unless `...` is included
```
</details>

## Why This Is A Bug

This violates the documented behavior of the `missing_dims` parameter. According to the docstring in `xarray/namedarray/core.py` (lines 1017-1022), when `missing_dims` is set to:
- `"ignore"`: The method should silently ignore missing dimensions
- `"warn"`: The method should issue a warning and continue, ignoring missing dimensions
- `"raise"`: The method should raise an exception (default behavior)

The bug occurs because the validation logic in `xarray/namedarray/utils.py` (line 172) checks whether the filtered dimensions form a complete permutation of all dimensions, regardless of the `missing_dims` setting. The `drop_missing_dims` function correctly filters out the missing dimension 'z' when `missing_dims='ignore'`, leaving only ('x',). However, the subsequent check `if set(existing_dims) ^ set(dims_all)` evaluates to True because ('x',) is not a complete permutation of ('x', 'y'), causing the function to raise a ValueError even though the user explicitly requested to ignore missing dimensions.

## Relevant Context

The bug is located in the `infix_dims` function in `/home/npc/miniconda/lib/python3.13/site-packages/xarray/namedarray/utils.py` at lines 170-176. The function correctly calls `drop_missing_dims` to filter out invalid dimensions based on the `missing_dims` parameter, but then unconditionally checks if the remaining dimensions form a complete permutation.

The `drop_missing_dims` function (lines 106-145 in the same file) correctly implements the three behaviors:
- For "raise": Raises ValueError for invalid dimensions
- For "warn": Issues a warning and filters out invalid dimensions
- For "ignore": Silently filters out invalid dimensions

The issue is that after filtering, the code still requires a complete permutation when no ellipsis is present, defeating the purpose of the `missing_dims` parameter.

Documentation: https://docs.xarray.dev/en/stable/generated/xarray.NamedArray.permute_dims.html

## Proposed Fix

```diff
--- a/xarray/namedarray/utils.py
+++ b/xarray/namedarray/utils.py
@@ -169,7 +169,8 @@ def infix_dims(
                 yield d
     else:
         existing_dims = drop_missing_dims(dims_supplied, dims_all, missing_dims)
-        if set(existing_dims) ^ set(dims_all):
+        # Only check for complete permutation if we're in 'raise' mode
+        if missing_dims == "raise" and set(existing_dims) ^ set(dims_all):
             raise ValueError(
                 f"{dims_supplied} must be a permuted list of {dims_all}, unless `...` is included"
             )
```