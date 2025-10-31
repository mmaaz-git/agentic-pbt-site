# Bug Report: xarray RangeIndex.arange Creates Negative Dimension Size

**Target**: `xarray.indexes.RangeIndex.arange`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`RangeIndex.arange` creates an index with negative dimension size when the step direction is incompatible with the start-to-stop direction, violating fundamental array constraints that dimensions must be non-negative.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from xarray.indexes import RangeIndex


@given(
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    step=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_arange_step_nonzero(start, stop, step):
    assume(step != 0)
    assume(abs(step) > 1e-10)
    assume((stop - start) / step < 1e6)
    assume((stop - start) / step > -1e6)

    index = RangeIndex.arange(
        start=start,
        stop=stop,
        step=step,
        dim="x"
    )
    assert index.size >= 0

# Run the test
test_arange_step_nonzero()
```

<details>

<summary>
**Failing input**: `start=1.0, stop=0.0, step=1.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 26, in <module>
    test_arange_step_nonzero()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 6, in test_arange_step_nonzero
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 23, in test_arange_step_nonzero
    assert index.size >= 0
           ^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_arange_step_nonzero(
    start=1.0,
    stop=0.0,
    step=1.0,
)
```
</details>

## Reproducing the Bug

```python
import math
import numpy as np
from xarray.indexes import RangeIndex

# Create RangeIndex with incompatible direction
index = RangeIndex.arange(start=1.0, stop=0.0, step=1.0, dim="x")
print(f"Size: {index.size}")
print(f"Start: {index.start}, Stop: {index.stop}, Step: {index.step}")

# Show how the size is computed
size_computed = math.ceil((0.0 - 1.0) / 1.0)
print(f"\nSize computed as: math.ceil((stop - start) / step) = math.ceil(({0.0} - {1.0}) / {1.0}) = {size_computed}")

# Compare with NumPy's behavior
print(f"\nNumPy's behavior:")
arr = np.arange(1.0, 0.0, 1.0)
print(f"np.arange(1.0, 0.0, 1.0) = {arr}")
print(f"np.arange(1.0, 0.0, 1.0).size = {arr.size}")

# Show that negative size is problematic
print(f"\nProblems with negative size:")
print(f"- A negative dimension size violates array constraints")
print(f"- Could cause downstream errors in operations expecting non-negative sizes")
print(f"- Deviates from NumPy's established behavior (returns empty array)")
```

<details>

<summary>
RangeIndex produces negative size (-1) instead of empty array like NumPy
</summary>
```
Size: -1
Start: 1.0, Stop: 0.0, Step: 1.0

Size computed as: math.ceil((stop - start) / step) = math.ceil((0.0 - 1.0) / 1.0) = -1

NumPy's behavior:
np.arange(1.0, 0.0, 1.0) = []
np.arange(1.0, 0.0, 1.0).size = 0

Problems with negative size:
- A negative dimension size violates array constraints
- Could cause downstream errors in operations expecting non-negative sizes
- Deviates from NumPy's established behavior (returns empty array)
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **Fundamental Array Constraint Violation**: Array dimensions must be non-negative integers. A negative size (-1) breaks this fundamental invariant that all array libraries rely on.

2. **NumPy Compatibility Break**: The method name `arange` and its signature directly mirror NumPy's `np.arange`, creating a strong expectation for compatible behavior. NumPy consistently returns an empty array (size 0) when step direction opposes the start-to-stop direction, never a negative size.

3. **Documentation Inconsistency**: The docstring states "End of interval. In general the interval does not include this value" but doesn't specify behavior for incompatible directions. Users would reasonably expect NumPy-compatible behavior or a clear error, not an invalid negative size.

4. **Potential for Downstream Errors**: Any code that uses this index's size for array allocation, iteration bounds, or mathematical operations will likely fail or produce incorrect results with a negative dimension size.

5. **Silent Data Integrity Issue**: The negative size doesn't immediately raise an error, allowing invalid state to propagate through the program until it causes problems elsewhere, making debugging difficult.

## Relevant Context

The bug occurs in `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/indexes/range_index.py:219`:

```python
size = math.ceil((stop - start) / step)
```

When `step > 0` but `start > stop` (or `step < 0` but `start < stop`), the division `(stop - start) / step` produces a negative value, which `math.ceil()` preserves as negative.

NumPy's implementation handles this by checking if the computed size would be negative and returning 0 instead. This is documented behavior in NumPy where "values are generated within the half-open interval [start, stop)" - if the interval is empty due to incompatible step direction, the result is an empty array.

Related code locations:
- Bug location: `xarray/indexes/range_index.py:219`
- NumPy reference: `numpy.arange` documentation and implementation

## Proposed Fix

```diff
--- a/xarray/indexes/range_index.py
+++ b/xarray/indexes/range_index.py
@@ -216,7 +216,7 @@ class RangeIndex(CoordinateTransformIndex):
         if coord_name is None:
             coord_name = dim

-        size = math.ceil((stop - start) / step)
+        size = max(0, math.ceil((stop - start) / step))

         transform = RangeCoordinateTransform(
             start, stop, size, coord_name, dim, dtype=dtype
```