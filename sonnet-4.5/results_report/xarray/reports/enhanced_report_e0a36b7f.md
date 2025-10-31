# Bug Report: xarray.indexes.RangeIndex.arange Step Parameter Not Preserved

**Target**: `xarray.indexes.RangeIndex.arange`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `RangeIndex.arange` method does not preserve the `step` parameter when creating a range index. Instead of using the provided step value directly, it recalculates the step as `(stop - start) / size`, resulting in different spacing between values than requested.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from xarray.indexes import RangeIndex
import numpy as np

@given(
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    step=st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=500)
def test_arange_step_matches_parameter(start, stop, step):
    assume(stop > start)
    assume(step > 0)
    assume((stop - start) / step < 1e6)

    index = RangeIndex.arange(start, stop, step, dim="x")

    coords = index.transform.forward({index.dim: np.arange(index.size)})
    values = coords[index.coord_name]

    if index.size > 1:
        actual_steps = np.diff(values)

        assert np.allclose(actual_steps, step, rtol=1e-9)

if __name__ == "__main__":
    test_arange_step_matches_parameter()
```

<details>

<summary>
**Failing input**: `start=0.0, stop=1.5, step=1.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 27, in <module>
    test_arange_step_matches_parameter()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 6, in test_arange_step_matches_parameter
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 24, in test_arange_step_matches_parameter
    assert np.allclose(actual_steps, step, rtol=1e-9)
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_arange_step_matches_parameter(
    start=0.0,  # or any other generated value
    stop=1.5,
    step=1.0,
)
```
</details>

## Reproducing the Bug

```python
from xarray.indexes import RangeIndex
import numpy as np

# Test the failing case
index = RangeIndex.arange(0.0, 1.5, 1.0, dim="x")

print(f"Expected step: 1.0")
print(f"Actual step: {index.step}")

coords = index.transform.forward({index.dim: np.arange(index.size)})
values = coords[index.coord_name]

print(f"Expected values: [0.0, 1.0]")
print(f"Actual values: {values}")

# Compare with numpy.arange
numpy_values = np.arange(0.0, 1.5, 1.0)
print(f"\nnumpy.arange(0.0, 1.5, 1.0) produces: {numpy_values}")
print(f"RangeIndex.arange produces different values: {values}")
```

<details>

<summary>
Output showing incorrect step value
</summary>
```
Expected step: 1.0
Actual step: 0.75
Expected values: [0.0, 1.0]
Actual values: [0.   0.75]

numpy.arange(0.0, 1.5, 1.0) produces: [0. 1.]
RangeIndex.arange produces different values: [0.   0.75]
```
</details>

## Why This Is A Bug

This violates the documented behavior and expected functionality in several critical ways:

1. **Documentation violation**: The docstring at line 155 in `/home/npc/miniconda/lib/python3.13/site-packages/xarray/indexes/range_index.py` explicitly states: "the index is within the half-open interval [start, stop), with spacing between values given by step". The actual implementation does not honor this promise.

2. **numpy.arange incompatibility**: The documentation (line 120) claims the methods are "similar to numpy.arange and numpy.linspace". However, `numpy.arange(0.0, 1.5, 1.0)` produces `[0.0, 1.0]` while `RangeIndex.arange(0.0, 1.5, 1.0)` produces `[0.0, 0.75]` - fundamentally different behavior.

3. **Example contradiction**: The documentation example (lines 190-201) shows `RangeIndex.arange(0.0, 1.0, 0.2, dim="x")` producing exact 0.2 spacing, implying step preservation. The actual implementation would not achieve this for many inputs.

4. **Silent data corruption**: The bug produces incorrect numerical results without any warning or error, potentially causing subtle bugs in downstream calculations.

## Relevant Context

The root cause is in the implementation at lines 219-223 of `range_index.py`:

1. Line 219: `size = math.ceil((stop - start) / step)` - Calculates the number of elements
2. Lines 221-223: Creates `RangeCoordinateTransform(start, stop, size, ...)`
3. Lines 58-65: The `step` property of `RangeCoordinateTransform` recalculates: `step = (stop - start) / size`

This means the provided `step` parameter is only used to determine the number of points, but the actual spacing is recalculated by evenly dividing the interval by that size. For example:
- Input: `start=0.0, stop=1.5, step=1.0`
- Calculated: `size = ceil(1.5 / 1.0) = 2`
- Recalculated step: `(1.5 - 0.0) / 2 = 0.75` (not the original 1.0)

This affects any use case where precise step values are required, which is a primary purpose of the `arange` function.

Documentation: https://docs.xarray.dev/en/stable/generated/xarray.indexes.RangeIndex.html
Source code: https://github.com/pydata/xarray/blob/main/xarray/indexes/range_index.py

## Proposed Fix

The fix requires preserving the original step value and using it consistently:

```diff
--- a/xarray/indexes/range_index.py
+++ b/xarray/indexes/range_index.py
@@ -219,8 +219,10 @@ class RangeIndex(CoordinateTransformIndex):
         size = math.ceil((stop - start) / step)

+        # Use actual endpoint based on step, not the provided stop
+        actual_stop = start + size * step
         transform = RangeCoordinateTransform(
-            start, stop, size, coord_name, dim, dtype=dtype
+            start, actual_stop, size, coord_name, dim, dtype=dtype
         )

         return cls(transform)
```

This ensures the step value is preserved exactly as provided, matching numpy.arange behavior and the documented contract.