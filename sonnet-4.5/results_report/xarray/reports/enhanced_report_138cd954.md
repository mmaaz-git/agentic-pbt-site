# Bug Report: xarray.structure.alignment.broadcast() String Exclude Parameter Incorrectly Excludes Dimensions

**Target**: `xarray.structure.alignment.broadcast`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `broadcast()` function incorrectly handles string values for the `exclude` parameter, treating the string as a sequence of characters rather than a single dimension name. This causes dimensions whose names are substrings of the exclude string to be incorrectly excluded from broadcasting.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import numpy as np
import xarray as xr
from xarray.structure.alignment import broadcast


@given(
    dim_name=st.text(min_size=2, max_size=5, alphabet=st.characters(whitelist_categories=('Ll',))),
    other_dim=st.text(min_size=1, max_size=1, alphabet=st.characters(whitelist_categories=('Ll',)))
)
@settings(max_examples=100)
def test_broadcast_exclude_string_exact_match(dim_name, other_dim):
    """Property: When exclude is a string dimension name, only that exact dimension should be excluded."""
    # Ensure other_dim is different from dim_name
    assume(dim_name != other_dim)

    # Create two DataArrays with different dimensions
    da1 = xr.DataArray([1, 2], dims=[dim_name])
    da2 = xr.DataArray([3, 4, 5], dims=[other_dim])

    # The bug occurs when other_dim is a substring (character) of dim_name
    # For example, if dim_name='xy' and other_dim='x', then 'x' in 'xy' is True
    # This causes dimension 'x' to be incorrectly excluded from broadcasting

    if other_dim in dim_name:  # This is when the bug triggers
        # Due to the bug, when we try to broadcast with exclude=dim_name,
        # the dimension other_dim will be incorrectly excluded
        try:
            result1, result2 = broadcast(da1, da2, exclude=dim_name)
            # If we get here, check if the bug manifested
            if other_dim not in result1.dims:
                print(f"BUG CONFIRMED: Dimension '{other_dim}' was incorrectly excluded")
                print(f"  exclude='{dim_name}' (a string)")
                print(f"  '{other_dim}' in '{dim_name}' = True (character-level check)")
                print(f"  Result1 dims: {result1.dims}, should have included '{other_dim}'")
                print(f"  Result2 dims: {result2.dims}")
                assert False, f"Bug: '{other_dim}' incorrectly excluded when exclude='{dim_name}'"
        except ValueError as e:
            # The bug can also manifest as a ValueError when dimensions are incorrectly excluded
            if "new dimensions" in str(e) and "must be a superset" in str(e):
                print(f"BUG CONFIRMED via ValueError: Dimension '{other_dim}' was incorrectly excluded")
                print(f"  exclude='{dim_name}' (a string)")
                print(f"  '{other_dim}' in '{dim_name}' = True (character-level check)")
                print(f"  Error: {e}")
                assert False, f"Bug caused ValueError: '{other_dim}' incorrectly excluded when exclude='{dim_name}'"
            else:
                raise  # Re-raise if it's a different error
    else:
        # When other_dim is not a substring of dim_name, broadcasting should work normally
        result1, result2 = broadcast(da1, da2, exclude=dim_name)
        assert other_dim in result1.dims, \
            f"Dimension '{other_dim}' should be in result1 after broadcasting"


if __name__ == "__main__":
    # Run the test to find a failing example
    try:
        test_broadcast_exclude_string_exact_match()
        print("Test completed without finding the bug (likely due to random generation)")
    except AssertionError as e:
        print(f"\nTest failed as expected, demonstrating the bug:")
        print(f"  {e}")
```

<details>

<summary>
**Failing input**: `dim_name='aa', other_dim='a'` (and many other combinations where one dimension name contains characters from another)
</summary>
```
BUG CONFIRMED via ValueError: Dimension 'a' was incorrectly excluded
  exclude='aa' (a string)
  'a' in 'aa' = True (character-level check)
  Error: new dimensions {} must be a superset of existing dimensions ('aa',)
BUG CONFIRMED via ValueError: Dimension 'a' was incorrectly excluded
  exclude='aa' (a string)
  'a' in 'aa' = True (character-level check)
  Error: new dimensions {} must be a superset of existing dimensions ('aa',)
BUG CONFIRMED via ValueError: Dimension 'a' was incorrectly excluded
  exclude='false' (a string)
  'a' in 'false' = True (character-level check)
  Error: new dimensions {} must be a superset of existing dimensions ('false',)
BUG CONFIRMED via ValueError: Dimension 'b' was incorrectly excluded
  exclude='űćwƚb' (a string)
  'b' in 'űćwƚb' = True (character-level check)
  Error: new dimensions {} must be a superset of existing dimensions ('űćwƚb',)
BUG CONFIRMED via ValueError: Dimension 'b' was incorrectly excluded
  exclude='ɕb𖹽ěᾕ' (a string)
  'b' in 'ɕb𖹽ěᾕ' = True (character-level check)
  Error: new dimensions {} must be a superset of existing dimensions ('ɕb𖹽ěᾕ',)
BUG CONFIRMED via ValueError: Dimension 'a' was incorrectly excluded
  exclude='aa' (a string)
  'a' in 'aa' = True (character-level check)
  Error: new dimensions {} must be a superset of existing dimensions ('aa',)
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 58, in <module>
  |     test_broadcast_exclude_string_exact_match()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 8, in test_broadcast_exclude_string_exact_match
  |     dim_name=st.text(min_size=2, max_size=5, alphabet=st.characters(whitelist_categories=('Ll',))),
  |                ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 50, in test_broadcast_exclude_string_exact_match
    |     result1, result2 = broadcast(da1, da2, exclude=dim_name)
    |                        ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/structure/alignment.py", line 1301, in broadcast
    |     result = [_broadcast_helper(arg, exclude, dims_map, common_coords) for arg in args]
    |               ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/structure/alignment.py", line 1169, in _broadcast_helper
    |     return _broadcast_array(arg)  # type: ignore[return-value,unused-ignore]
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/structure/alignment.py", line 1154, in _broadcast_array
    |     data = _set_dims(array.variable)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/structure/alignment.py", line 1151, in _set_dims
    |     return var.set_dims(var_dims_map)
    |            ~~~~~~~~~~~~^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/util/deprecation_helpers.py", line 144, in wrapper
    |     return func(*args, **kwargs)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/core/variable.py", line 1461, in set_dims
    |     raise ValueError(
    |     ...<2 lines>...
    |     )
    | ValueError: new dimensions {'b': 3} must be a superset of existing dimensions ('aa',)
    | Falsifying example: test_broadcast_exclude_string_exact_match(
    |     # The test always failed when commented parts were varied together.
    |     dim_name='aa',  # or any other generated value
    |     other_dim='b',  # or any other generated value
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 29, in test_broadcast_exclude_string_exact_match
    |     result1, result2 = broadcast(da1, da2, exclude=dim_name)
    |                        ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/structure/alignment.py", line 1301, in broadcast
    |     result = [_broadcast_helper(arg, exclude, dims_map, common_coords) for arg in args]
    |               ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/structure/alignment.py", line 1169, in _broadcast_helper
    |     return _broadcast_array(arg)  # type: ignore[return-value,unused-ignore]
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/structure/alignment.py", line 1154, in _broadcast_array
    |     data = _set_dims(array.variable)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/structure/alignment.py", line 1151, in _set_dims
    |     return var.set_dims(var_dims_map)
    |            ~~~~~~~~~~~~^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/util/deprecation_helpers.py", line 144, in wrapper
    |     return func(*args, **kwargs)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/core/variable.py", line 1461, in set_dims
    |     raise ValueError(
    |     ...<2 lines>...
    |     )
    | ValueError: new dimensions {} must be a superset of existing dimensions ('aa',)
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 45, in test_broadcast_exclude_string_exact_match
    |     assert False, f"Bug caused ValueError: '{other_dim}' incorrectly excluded when exclude='{dim_name}'"
    |            ^^^^^
    | AssertionError: Bug caused ValueError: 'a' incorrectly excluded when exclude='aa'
    | Falsifying example: test_broadcast_exclude_string_exact_match(
    |     dim_name='aa',
    |     other_dim='a',
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import numpy as np
import xarray as xr
from xarray.structure.alignment import broadcast

# Create two DataArrays with different dimensions
da_x = xr.DataArray([1, 2, 3], dims=['x'])
da_y = xr.DataArray([4, 5], dims=['y'])

# Try to broadcast with exclude='xy' (a string that doesn't match any dimension)
# Expected: dimensions 'x' and 'y' should be broadcast normally since 'xy' doesn't exist
# Actual: Both 'x' and 'y' are excluded because Python checks 'x' in 'xy' and 'y' in 'xy'
result_x, result_y = broadcast(da_x, da_y, exclude='xy')

print(f"Original da_x dims: {da_x.dims}")
print(f"Original da_y dims: {da_y.dims}")
print(f"Result da_x dims after broadcast: {result_x.dims}")
print(f"Result da_y dims after broadcast: {result_y.dims}")
print()

# Check if dimensions were incorrectly excluded
if 'x' not in result_y.dims:
    print("ERROR: Dimension 'x' was incorrectly excluded from da_y!")
    print("  This happened because 'x' in 'xy' returns True (character membership)")
else:
    print("Dimension 'x' was correctly broadcast to da_y")

if 'y' not in result_x.dims:
    print("ERROR: Dimension 'y' was incorrectly excluded from da_x!")
    print("  This happened because 'y' in 'xy' returns True (character membership)")
else:
    print("Dimension 'y' was correctly broadcast to da_x")

print()
print("Demonstrating the root cause:")
print(f"  'x' in 'xy' = {'x' in 'xy'}  # Should check dimension name 'xy', not character 'x'")
print(f"  'y' in 'xy' = {'y' in 'xy'}  # Should check dimension name 'xy', not character 'y'")
```

<details>

<summary>
Output demonstrating incorrect dimension exclusion
</summary>
```
Original da_x dims: ('x',)
Original da_y dims: ('y',)
Result da_x dims after broadcast: ('x',)
Result da_y dims after broadcast: ('y',)

ERROR: Dimension 'x' was incorrectly excluded from da_y!
  This happened because 'x' in 'xy' returns True (character membership)
ERROR: Dimension 'y' was incorrectly excluded from da_x!
  This happened because 'y' in 'xy' returns True (character membership)

Demonstrating the root cause:
  'x' in 'xy' = True  # Should check dimension name 'xy', not character 'x'
  'y' in 'xy' = True  # Should check dimension name 'xy', not character 'y'
```
</details>

## Why This Is A Bug

This behavior violates expected functionality and the documented interface in several ways:

1. **Type Contract Violation**: The function signature explicitly accepts `exclude: str | Iterable[Hashable] | None`, documenting that a string can be passed. The type hint indicates a string should represent a single dimension name to exclude, not a sequence of characters.

2. **Inconsistency with align()**: The `align()` function in the same module (`xarray.structure.alignment`) correctly handles string exclude parameters by converting them to a list (lines 212-214 of alignment.py). The `broadcast()` function should behave consistently.

3. **Counterintuitive Behavior**: No reasonable user would expect `exclude='xy'` to exclude both 'x' and 'y' dimensions. The string 'xy' should refer to a single dimension named 'xy', not all dimensions whose names are characters within that string.

4. **Python String Membership Semantics**: The bug occurs because Python's `in` operator treats strings as sequences of characters. When the code checks `dim not in exclude` (line 1129) or iterates with `for dim in exclude` (line 1146), it's performing character-level operations rather than checking dimension names.

5. **Silent Incorrect Behavior**: The bug can cause dimensions to be silently excluded from broadcasting, leading to unexpected data shapes and potentially incorrect calculations downstream.

## Relevant Context

The bug is located in `/home/npc/miniconda/lib/python3.13/site-packages/xarray/structure/alignment.py`:

- Line 1298: `broadcast()` passes the string exclude directly to `align()`
- Line 1300: The string is passed to `_get_broadcast_dims_map_common_coords()`
- Line 1129: `if dim not in common_coords and dim not in exclude:` - performs character check
- Line 1146: `for dim in exclude:` - iterates over characters instead of dimension names

The `Aligner` class used by `align()` correctly handles this case (lines 212-214):
```python
if isinstance(exclude_dims, str):
    exclude_dims = [exclude_dims]
self.exclude_dims = frozenset(exclude_dims)
```

Documentation: https://docs.xarray.dev/en/stable/generated/xarray.broadcast.html

## Proposed Fix

```diff
--- a/xarray/structure/alignment.py
+++ b/xarray/structure/alignment.py
@@ -1295,7 +1295,11 @@ def broadcast(

     if exclude is None:
         exclude = set()
-    args = align(*args, join="outer", copy=False, exclude=exclude)
+    elif isinstance(exclude, str):
+        # Convert string to a set containing the single dimension name
+        # to avoid treating it as a sequence of characters
+        exclude = {exclude}
+    args = align(*args, join="outer", copy=False, exclude=exclude)

     dims_map, common_coords = _get_broadcast_dims_map_common_coords(args, exclude)
     result = [_broadcast_helper(arg, exclude, dims_map, common_coords) for arg in args]
```