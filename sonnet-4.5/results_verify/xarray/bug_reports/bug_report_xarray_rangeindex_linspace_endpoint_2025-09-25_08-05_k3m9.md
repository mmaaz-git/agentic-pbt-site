# Bug Report: xarray RangeIndex linspace endpoint precision

**Target**: `xarray.indexes.RangeIndex.linspace`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`RangeIndex.linspace()` with `endpoint=True` does not guarantee that the last generated value exactly equals the specified `stop` parameter due to floating-point error accumulation in the value generation formula.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import numpy as np
from xarray.indexes import RangeIndex

@given(
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    num=st.integers(min_value=2, max_value=1000),
)
def test_linspace_endpoint_true_last_value_equals_stop(start, stop, num):
    assume(abs(stop - start) > 1e-6)

    index = RangeIndex.linspace(start, stop, num, endpoint=True, dim="x")
    values = index.transform.forward({"x": np.arange(num)})["x"]

    assert values[-1] == stop
```

**Failing input**: `start=817040.0, stop=0.0, num=18`

## Reproducing the Bug

```python
import numpy as np
from xarray.indexes import RangeIndex

start, stop, num = 817040.0, 0.0, 18

index = RangeIndex.linspace(start, stop, num, endpoint=True, dim="x")
values = index.transform.forward({"x": np.arange(num)})["x"]

print(f"Last value: {values[-1]}")
print(f"Expected:   {stop}")
print(f"Match: {values[-1] == stop}")

numpy_comparison = np.linspace(start, stop, num, endpoint=True)
print(f"\nnumpy.linspace last value: {numpy_comparison[-1]}")
print(f"numpy match: {numpy_comparison[-1] == stop}")
```

Output:
```
Last value: 1.1641532182693481e-10
Expected:   0.0
Match: False

numpy.linspace last value: 0.0
numpy match: True
```

## Why This Is A Bug

The docstring for `RangeIndex.linspace()` states that when `endpoint=True`, "the `stop` value is included in the interval" (line 251 in range_index.py). This is standard behavior matching `numpy.linspace()`, which guarantees exact endpoint values.

The root cause is the use of a numerically unstable formula in `RangeCoordinateTransform.forward()` (line 69):

```python
labels = self.start + positions * self.step
```

When computing the last value with `position = num-1`, floating-point errors accumulate, preventing the result from exactly equaling `stop`. NumPy avoids this issue by using a more stable formula that interpolates between start and stop:

```python
# NumPy's approach (simplified)
value[i] = start * (1 - i/(num-1)) + stop * (i/(num-1))
```

This ensures `value[0] = start` and `value[num-1] = stop` exactly, without floating-point drift.

## Fix

The fix should modify `RangeCoordinateTransform.forward()` to use a numerically stable endpoint-preserving formula when the index was created with `endpoint=True`. However, `RangeCoordinateTransform` currently doesn't track whether it was created with `endpoint=True`, so this would require architectural changes.

A simpler fix is to ensure the last value is exactly `stop` when appropriate:

```diff
--- a/xarray/indexes/range_index.py
+++ b/xarray/indexes/range_index.py
@@ -66,7 +66,11 @@ class RangeCoordinateTransform(CoordinateTransform):

     def forward(self, dim_positions: dict[str, Any]) -> dict[Hashable, Any]:
         positions = dim_positions[self.dim]
-        labels = self.start + positions * self.step
+        positions_arr = np.asarray(positions)
+        # Use numerically stable formula to preserve endpoints
+        t = positions_arr / self.size if self.size > 0 else positions_arr
+        labels = self.start * (1 - t) + self.stop * t
         return {self.coord_name: labels}
```

This formula ensures that:
- When `positions = 0`: `labels = start * 1 + stop * 0 = start`
- When `positions = size`: `labels = start * 0 + stop * 1 = stop`