# Bug Report: xarray RangeIndex linspace endpoint precision error

**Target**: `xarray.indexes.RangeIndex.linspace`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`RangeIndex.linspace()` with `endpoint=True` fails to guarantee that the last generated value exactly equals the specified `stop` parameter due to floating-point error accumulation in the coordinate generation formula, violating its documented behavior and diverging from numpy.linspace's standard.

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

<details>

<summary>
**Failing input**: `start=1000000.0, stop=-48576.95854238502, num=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 20, in <module>
    test_linspace_endpoint_true_last_value_equals_stop()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 6, in test_linspace_endpoint_true_last_value_equals_stop
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 16, in test_linspace_endpoint_true_last_value_equals_stop
    assert values[-1] == stop
           ^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_linspace_endpoint_true_last_value_equals_stop(
    start=1000000.0,
    stop=-48576.95854238502,
    num=2,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from xarray.indexes import RangeIndex

# Failing case from the bug report
start, stop, num = 817040.0, 0.0, 18

# Create xarray RangeIndex with endpoint=True
index = RangeIndex.linspace(start, stop, num, endpoint=True, dim="x")

# Get the actual values
values = index.transform.forward({"x": np.arange(num)})["x"]

print("=== XArray RangeIndex.linspace ===")
print(f"Start: {start}")
print(f"Stop: {stop}")
print(f"Number of points: {num}")
print(f"Endpoint: True")
print()
print(f"Last value generated: {values[-1]}")
print(f"Expected last value:  {stop}")
print(f"Exact match (values[-1] == stop): {values[-1] == stop}")
print(f"Error: {abs(values[-1] - stop)}")
print(f"Relative error: {abs((values[-1] - stop) / start) if start != 0 else 'N/A'}")
print()

# Compare with numpy.linspace
numpy_values = np.linspace(start, stop, num, endpoint=True)
print("=== NumPy linspace comparison ===")
print(f"NumPy last value: {numpy_values[-1]}")
print(f"NumPy exact match (last == stop): {numpy_values[-1] == stop}")
print()

# Show all values for context
print("=== All XArray values ===")
for i, val in enumerate(values):
    print(f"  [{i:2d}]: {val}")
```

<details>

<summary>
Last value is 1.16e-10 instead of expected 0.0
</summary>
```
=== XArray RangeIndex.linspace ===
Start: 817040.0
Stop: 0.0
Number of points: 18
Endpoint: True

Last value generated: 1.1641532182693481e-10
Expected last value:  0.0
Exact match (values[-1] == stop): False
Error: 1.1641532182693481e-10
Relative error: 1.4248423801397093e-16

=== NumPy linspace comparison ===
NumPy last value: 0.0
NumPy exact match (last == stop): True

=== All XArray values ===
  [ 0]: 817040.0
  [ 1]: 768978.8235294118
  [ 2]: 720917.6470588235
  [ 3]: 672856.4705882353
  [ 4]: 624795.2941176471
  [ 5]: 576734.1176470588
  [ 6]: 528672.9411764706
  [ 7]: 480611.7647058824
  [ 8]: 432550.58823529416
  [ 9]: 384489.4117647059
  [10]: 336428.2352941177
  [11]: 288367.0588235295
  [12]: 240305.8823529412
  [13]: 192244.705882353
  [14]: 144183.52941176482
  [15]: 96122.3529411765
  [16]: 48061.17647058831
  [17]: 1.1641532182693481e-10
```
</details>

## Why This Is A Bug

This violates the documented behavior and user expectations in several critical ways:

1. **Documentation Contract Violation**: The docstring at line 250-251 of `range_index.py` explicitly states that when `endpoint=True`, "the `stop` value is included in the interval." In numerical computing, "included" means exactly equal, not approximately equal.

2. **Inconsistency with NumPy**: The function is explicitly documented (lines 119-120) as being "similar to :py:func:`numpy.linspace`". NumPy's linspace guarantees exact endpoint values, as demonstrated in the reproduction where `numpy.linspace(817040.0, 0.0, 18)[-1] == 0.0` returns `True`.

3. **Numerical Instability**: The root cause is in `RangeCoordinateTransform.forward()` (line 69) which uses the formula `labels = self.start + positions * self.step`. This accumulates floating-point errors, especially when `positions` is large (e.g., 17 in our case).

4. **Breaking Equality Checks**: Scientific computing code often relies on exact equality checks for boundary conditions. Code like `if values[-1] == stop` will fail unexpectedly, potentially causing subtle bugs in downstream calculations.

5. **Precision Degradation**: While the error seems small (1.16e-10), it represents a failure of the fundamental promise of the function. The relative error (~1.42e-16) is within machine epsilon but still incorrect when exact values are expected and achievable.

## Relevant Context

The issue stems from how `RangeIndex.linspace` adjusts the stop value when `endpoint=True` (lines 282-283):
```python
if endpoint:
    stop += (stop - start) / (num - 1)
```

This adjustment, combined with the step-based formula in `forward()`, causes precision loss. NumPy avoids this by using interpolation:
```python
# Simplified NumPy approach
value[i] = start * (1 - i/(num-1)) + stop * (i/(num-1))
```

This guarantees `value[0] = start` and `value[num-1] = stop` exactly without accumulation errors.

Key code locations:
- Bug location: `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/indexes/range_index.py:69`
- Problematic adjustment: `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/indexes/range_index.py:282-283`
- Documentation: `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/indexes/range_index.py:250-251`

## Proposed Fix

The fix requires modifying `RangeCoordinateTransform` to use a numerically stable interpolation formula when exact endpoints are needed. However, since `RangeCoordinateTransform` doesn't currently track whether it was created with `endpoint=True`, a minimal fix could special-case the last position:

```diff
--- a/xarray/indexes/range_index.py
+++ b/xarray/indexes/range_index.py
@@ -66,7 +66,16 @@ class RangeCoordinateTransform(CoordinateTransform):

     def forward(self, dim_positions: dict[str, Any]) -> dict[Hashable, Any]:
         positions = dim_positions[self.dim]
-        labels = self.start + positions * self.step
+        positions_arr = np.asarray(positions)
+
+        # Use standard formula for most values
+        labels = self.start + positions_arr * self.step
+
+        # Ensure exact endpoint when position equals size-1
+        # This handles the endpoint=True case from linspace
+        mask = positions_arr == (self.size - 1)
+        if np.any(mask):
+            labels = np.where(mask, self.stop - self.step, labels)
+
         return {self.coord_name: labels}
```

A more robust solution would use interpolation for all values when created via `linspace` with `endpoint=True`, similar to NumPy's approach, but would require tracking the creation method in the transform object.