# Bug Report: xarray RangeCoordinateTransform.reverse Division by Zero Returns NaN

**Target**: `xarray.indexes.range_index.RangeCoordinateTransform.reverse`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `RangeCoordinateTransform.reverse()` method returns NaN when `start == stop` due to division by zero, violating the documented forward/reverse round-trip property that these methods should be inverse transformations.

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
    ), f"Round-trip failed: {original_labels[transform.coord_name]} != {reconstructed_labels[transform.coord_name]}"

if __name__ == "__main__":
    test_range_transform_forward_reverse_roundtrip()
```

<details>

<summary>
**Failing input**: `start=0.0, stop=0.0, size=1, position=0`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/xarray/indexes/range_index.py:74: RuntimeWarning: invalid value encountered in divide
  positions = (labels - self.start) / self.step
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 36, in <module>
    test_range_transform_forward_reverse_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 7, in test_range_transform_forward_reverse_roundtrip
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 28, in test_range_transform_forward_reverse_roundtrip
    assert np.allclose(
           ~~~~~~~~~~~^
        original_labels[transform.coord_name],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        atol=1e-10
        ^^^^^^^^^^
    ), f"Round-trip failed: {original_labels[transform.coord_name]} != {reconstructed_labels[transform.coord_name]}"
    ^
AssertionError: Round-trip failed: [0.] != [nan]
Falsifying example: test_range_transform_forward_reverse_roundtrip(
    start=0.0,
    stop=0.0,
    size=1,  # or any other generated value
    position=0,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/59/hypo.py:33
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:919
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1085
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1095
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1708
```
</details>

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

<details>

<summary>
RuntimeWarning and NaN output
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/xarray/indexes/range_index.py:74: RuntimeWarning: invalid value encountered in divide
  positions = (labels - self.start) / self.step
step: 0.0
Original labels: [0.]
After round-trip: [nan]
```
</details>

## Why This Is A Bug

This violates the fundamental contract of coordinate transformations as documented in the `CoordinateTransform` abstract base class. The documentation explicitly states that `forward()` performs "grid -> world coordinate transformation" and `reverse()` performs "world -> grid coordinate reverse transformation", establishing them as inverse operations.

When `start == stop`, the `step` property calculates `(stop - start) / size = 0 / size = 0.0`. This causes the `reverse()` method at line 74 to perform division by zero: `positions = (labels - self.start) / self.step`. This produces NaN values instead of valid grid positions, breaking the mathematical property that `forward(reverse(labels))` should equal `labels`.

A constant coordinate transform (where all grid positions map to the same coordinate value) is mathematically valid and represents a degenerate but legitimate coordinate system. The constructor accepts `start == stop` without error, and the `forward()` method handles `step=0` correctly by mapping all positions to the same constant value. However, the `reverse()` method fails to handle this case properly.

## Relevant Context

- The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/xarray/indexes/range_index.py` at line 74
- The `RangeCoordinateTransform` class inherits from `CoordinateTransform` which documents the forward/reverse methods as inverse transformations
- The class already handles edge cases like `size=0` (returning `step=1.0` in line 65), showing that edge cases are considered in the design
- This affects both `RangeIndex.arange()` and `RangeIndex.linspace()` methods when they create constant coordinates
- The bug produces user-visible RuntimeWarnings about "invalid value encountered in divide"

## Proposed Fix

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