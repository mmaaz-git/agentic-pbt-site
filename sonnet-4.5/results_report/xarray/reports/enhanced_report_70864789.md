# Bug Report: xarray RangeIndex.arange Produces Negative Dimension Size

**Target**: `xarray.indexes.range_index.RangeIndex.arange`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `RangeIndex.arange` method produces an index with negative dimension size when the step direction doesn't allow traversal from start to stop, violating the fundamental invariant that array dimensions must be non-negative.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test for RangeIndex.arange that discovered the negative size bug.
"""

import sys
import os

# Add the xarray environment to path
sys.path.insert(0, '/home/npc/miniconda/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume, settings
from xarray.indexes.range_index import RangeIndex

@given(
    start=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    stop=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    step=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
)
@settings(max_examples=1000)
def test_arange_size_nonnegative(start, stop, step):
    assume(step != 0)
    assume(abs(step) > 1e-10)

    index = RangeIndex.arange(start, stop, step, dim="x")
    assert index.size >= 0, f"Size must be non-negative, got {index.size}"

# Run the test
if __name__ == "__main__":
    print("Running property-based test for RangeIndex.arange...")
    print()

    try:
        test_arange_size_nonnegative()
        print("All tests passed!")
    except AssertionError as e:
        print("Test failed!")
        print()
        print("The test found a case where RangeIndex.arange produces a negative size.")
        print("This violates the invariant that array dimensions must be non-negative.")
        print()
        print("Minimal failing example found by Hypothesis:")
        print("  start=1.0, stop=0.0, step=1.0")
        print()
        print("Running the minimal example directly...")
        index = RangeIndex.arange(1.0, 0.0, 1.0, dim="x")
        print(f"  Result: index.size = {index.size} (should be >= 0)")
        print()
        print("This proves that RangeIndex.arange can produce negative dimension sizes,")
        print("which violates fundamental array invariants.")
```

<details>

<summary>
**Failing input**: `start=1.0, stop=0.0, step=1.0`
</summary>
```
Running property-based test for RangeIndex.arange...

Test failed!

The test found a case where RangeIndex.arange produces a negative size.
This violates the invariant that array dimensions must be non-negative.

Minimal failing example found by Hypothesis:
  start=1.0, stop=0.0, step=1.0

Running the minimal example directly...
  Result: index.size = -1 (should be >= 0)

This proves that RangeIndex.arange can produce negative dimension sizes,
which violates fundamental array invariants.
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of the RangeIndex negative size bug.
"""

import sys
import os
import numpy as np

# Add the xarray environment to path
sys.path.insert(0, '/home/npc/miniconda/lib/python3.13/site-packages')

from xarray.indexes.range_index import RangeIndex
import xarray as xr

print("=== Testing RangeIndex.arange with mismatched step direction ===")
print()

# The problematic case: positive step going from 1.0 to 0.0
print("Test case: RangeIndex.arange(1.0, 0.0, 1.0, dim='x')")
print("Expected: size should be 0 (empty range)")
print()

# Create the index
index = RangeIndex.arange(1.0, 0.0, 1.0, dim="x")

# Show the problematic negative size
print(f"Actual result:")
print(f"  index.size = {index.size}")
print(f"  index.start = {index.start}")
print(f"  index.stop = {index.stop}")
print(f"  index.step = {index.step}")
print()

# Compare with NumPy's behavior
print("NumPy comparison:")
np_result = np.arange(1.0, 0.0, 1.0)
print(f"  np.arange(1.0, 0.0, 1.0) = {np_result}")
print(f"  np.arange(1.0, 0.0, 1.0).size = {np_result.size}")
print()

# Try to use this index in an xarray Dataset
print("=== Creating xarray Dataset with the negative-sized index ===")
try:
    coords = xr.Coordinates.from_xindex(index)
    ds = xr.Dataset(coords=coords)
    print(f"Dataset created: {ds}")
    print()

    # Try to add a data variable - this will fail due to negative dimension
    print("Attempting to add a data variable...")
    ds["temperature"] = xr.DataArray(np.zeros(abs(index.size)), dims=["x"])
    print("Success (should not happen)")

except Exception as e:
    print(f"Error when using negative-sized index: {e}")
    print(f"Error type: {type(e).__name__}")
```

<details>

<summary>
AlignmentError when attempting to use dataset with negative dimension
</summary>
```
=== Testing RangeIndex.arange with mismatched step direction ===

Test case: RangeIndex.arange(1.0, 0.0, 1.0, dim='x')
Expected: size should be 0 (empty range)

Actual result:
  index.size = -1
  index.start = 1.0
  index.stop = 0.0
  index.step = 1.0

NumPy comparison:
  np.arange(1.0, 0.0, 1.0) = []
  np.arange(1.0, 0.0, 1.0).size = 0

=== Creating xarray Dataset with the negative-sized index ===
Dataset created: <xarray.Dataset> Size: -8B
Dimensions:  (x: -1)
Coordinates:
  * x        (x) float64 -8B
Data variables:
    *empty*
Indexes:
    x        RangeIndex (start=1, stop=0, step=1)

Attempting to add a data variable...
Error when using negative-sized index: cannot reindex or align along dimension 'x' because of conflicting dimension sizes: {1, -1} (note: an index is found along that dimension with size=-1)
Error type: AlignmentError
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Fundamental Invariant Violation**: Array dimensions must always be non-negative. A dimension size of -1 is mathematically invalid and breaks core assumptions throughout the xarray codebase. The AlignmentError demonstrates that downstream code cannot handle negative dimensions.

2. **NumPy Incompatibility**: The method name `arange` and documentation strongly imply NumPy-compatible behavior. NumPy's `arange(1.0, 0.0, 1.0)` correctly returns an empty array with size 0, not a negative size. Users familiar with NumPy would expect identical behavior.

3. **Mathematical Incorrectness**: When using a positive step (1.0) to traverse from 1.0 to 0.0, it's impossible to reach the stop value. The mathematically correct result is an empty range (size=0), not a negative size.

4. **Runtime Failures**: The negative size causes actual failures when using the dataset. The AlignmentError shows that xarray's own code expects non-negative dimensions and fails when this invariant is violated.

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/xarray/indexes/range_index.py` at line 219:

```python
size = math.ceil((stop - start) / step)
```

When `start=1.0, stop=0.0, step=1.0`, this calculation yields:
- `(0.0 - 1.0) / 1.0 = -1.0`
- `math.ceil(-1.0) = -1`

The formula doesn't account for cases where the step direction prevents reaching the stop value from the start value. This affects any call where:
- `step > 0` and `stop < start` (trying to go backward with positive step)
- `step < 0` and `stop > start` (trying to go forward with negative step)

Documentation reference: https://docs.xarray.dev/en/stable/generated/xarray.indexes.RangeIndex.html#xarray.indexes.RangeIndex.arange

The documentation states the index is created within a half-open interval [start, stop) but doesn't specify behavior when the interval cannot be traversed with the given step.

## Proposed Fix

```diff
--- a/xarray/indexes/range_index.py
+++ b/xarray/indexes/range_index.py
@@ -216,7 +216,14 @@ class RangeIndex(CoordinateTransformIndex):
         if coord_name is None:
             coord_name = dim

-        size = math.ceil((stop - start) / step)
+        # Calculate size, ensuring it's non-negative
+        # Empty range when step direction doesn't match interval direction
+        raw_size = (stop - start) / step
+        if raw_size < 0:
+            # Step goes wrong direction - empty range
+            size = 0
+        else:
+            size = math.ceil(raw_size)

         transform = RangeCoordinateTransform(
             start, stop, size, coord_name, dim, dtype=dtype
```