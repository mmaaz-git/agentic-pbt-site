# Bug Report: xarray RangeCoordinateTransform Reverse Returns NaN

**Target**: `xarray.indexes.range_index.RangeCoordinateTransform.reverse`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`RangeCoordinateTransform.reverse` returns NaN when `start == stop`, violating the forward/reverse round-trip property.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from xarray.indexes.range_index import RangeCoordinateTransform


@given(
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    size=st.integers(min_value=1, max_value=1000),
    position=st.integers(min_value=0, max_value=999)
)
@settings(max_examples=200)
def test_range_transform_forward_reverse_roundtrip(start, stop, size, position):
    assume(position < size)

    transform = RangeCoordinateTransform(
        start=start,
        stop=stop,
        size=size,
        coord_name="x",
        dim="x",
    )

    original_labels = {transform.coord_name: np.array([start + position * transform.step])}
    positions = transform.reverse(original_labels)
    reconstructed_labels = transform.forward(positions)

    assert np.allclose(
        original_labels[transform.coord_name],
        reconstructed_labels[transform.coord_name],
        rtol=1e-10,
        atol=1e-10
    )
```

**Failing input**: `start=0.0, stop=0.0, size=1, position=0`

## Reproducing the Bug

```python
import numpy as np
from xarray.indexes.range_index import RangeCoordinateTransform

transform = RangeCoordinateTransform(
    start=0.0,
    stop=0.0,
    size=1,
    coord_name="x",
    dim="x",
)

print(f"step: {transform.step}")

original_labels = {transform.coord_name: np.array([0.0])}
positions = transform.reverse(original_labels)
reconstructed_labels = transform.forward(positions)

print(f"Original labels: {original_labels[transform.coord_name]}")
print(f"After round-trip: {reconstructed_labels[transform.coord_name]}")
```

Output:
```
step: 0.0
Original labels: [0.]
After round-trip: [nan]
```

The issue occurs at `range_index.py:74`:
```python
def reverse(self, coord_labels: dict[Hashable, Any]) -> dict[str, Any]:
    labels = coord_labels[self.coord_name]
    positions = (labels - self.start) / self.step  # Division by zero when step=0
    return {self.dim: positions}
```

When `start == stop`, `step` is `(stop - start) / size = 0.0`, causing division by zero.

## Why This Is A Bug

The forward/reverse methods are documented as coordinate transforms, implying they should be inverses of each other. When `start == stop`, the transform represents a constant coordinate value, and the reverse transformation should return the position(s) that map to that constant value. Instead, it returns NaN, violating the round-trip property.

## Fix

```diff
--- a/xarray/indexes/range_index.py
+++ b/xarray/indexes/range_index.py
@@ -72,7 +72,11 @@ class RangeCoordinateTransform(CoordinateTransform):

     def reverse(self, coord_labels: dict[Hashable, Any]) -> dict[str, Any]:
         labels = coord_labels[self.coord_name]
-        positions = (labels - self.start) / self.step
+        if self.step == 0:
+            # When start == stop, all positions map to the same label
+            positions = np.zeros_like(labels)
+        else:
+            positions = (labels - self.start) / self.step
         return {self.dim: positions}

     def equals(
```