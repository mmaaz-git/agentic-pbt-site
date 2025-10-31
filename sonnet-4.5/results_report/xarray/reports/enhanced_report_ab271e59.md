# Bug Report: xarray.plot.utils._rescale_imshow_rgb Missing Input Validation for vmin/vmax Bounds

**Target**: `xarray.plot.utils._rescale_imshow_rgb`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_rescale_imshow_rgb` function fails to validate that `vmin < vmax` when both parameters are explicitly provided, leading to mathematically incorrect scaling and silent data corruption in RGB image visualization.

## Property-Based Test

```python
from hypothesis import given, assume, strategies as st
import numpy as np
from xarray.plot.utils import _rescale_imshow_rgb
import pytest

@given(
    st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e6),
    st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e6)
)
def test_rescale_imshow_rgb_vmin_greater_than_vmax_rejected(vmin, vmax):
    assume(vmin > vmax)
    darray = np.random.uniform(0, 100, (10, 10, 3)).astype('f8')

    with pytest.raises(ValueError):
        _rescale_imshow_rgb(darray, vmin=vmin, vmax=vmax, robust=False)

# Run the test
if __name__ == "__main__":
    test_rescale_imshow_rgb_vmin_greater_than_vmax_rejected()
```

<details>

<summary>
**Failing input**: `vmin=1.0, vmax=0.0`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/40
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_rescale_imshow_rgb_vmin_greater_than_vmax_rejected FAILED  [100%]

=================================== FAILURES ===================================
___________ test_rescale_imshow_rgb_vmin_greater_than_vmax_rejected ____________

    @given(
>       st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e6),
                   ^^^
        st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e6)
    )

hypo.py:7:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

vmin = 1.0, vmax = 0.0

    @given(
        st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e6),
        st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e6)
    )
    def test_rescale_imshow_rgb_vmin_greater_than_vmax_rejected(vmin, vmax):
        assume(vmin > vmax)
        darray = np.random.uniform(0, 100, (10, 10, 3)).astype('f8')

>       with pytest.raises(ValueError):
             ^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE <class 'ValueError'>
E       Falsifying example: test_rescale_imshow_rgb_vmin_greater_than_vmax_rejected(
E           vmin=1.0,  # or any other generated value
E           vmax=0.0,
E       )

hypo.py:14: Failed
=============================== warnings summary ===============================
hypo.py::test_rescale_imshow_rgb_vmin_greater_than_vmax_rejected
hypo.py::test_rescale_imshow_rgb_vmin_greater_than_vmax_rejected
hypo.py::test_rescale_imshow_rgb_vmin_greater_than_vmax_rejected
hypo.py::test_rescale_imshow_rgb_vmin_greater_than_vmax_rejected
  /home/npc/miniconda/lib/python3.13/site-packages/xarray/plot/utils.py:778: RuntimeWarning: overflow encountered in cast
    darray = ((darray.astype("f8") - vmin) / (vmax - vmin)).astype("f4")

hypo.py::test_rescale_imshow_rgb_vmin_greater_than_vmax_rejected
hypo.py::test_rescale_imshow_rgb_vmin_greater_than_vmax_rejected
  /home/npc/miniconda/lib/python3.13/site-packages/xarray/plot/utils.py:778: RuntimeWarning: overflow encountered in divide
    darray = ((darray.astype("f8") - vmin) / (vmax - vmin)).astype("f4")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED hypo.py::test_rescale_imshow_rgb_vmin_greater_than_vmax_rejected - Fai...
======================== 1 failed, 6 warnings in 0.75s =========================
```
</details>

## Reproducing the Bug

```python
import numpy as np
from xarray.plot.utils import _rescale_imshow_rgb

# Test case 1: vmin > vmax (should raise ValueError but doesn't)
darray = np.array([[[50.0, 50.0, 50.0]]]).astype('f8')

print("Test 1: vmin=100.0, vmax=0.0")
print("Expected: ValueError")
print("Actual:")
try:
    result = _rescale_imshow_rgb(darray, vmin=100.0, vmax=0.0, robust=False)
    print(f"No error raised! Result: {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")

print("\n" + "="*50 + "\n")

# Test case 2: vmin == vmax (should raise ValueError but causes division by zero)
print("Test 2: vmin=50.0, vmax=50.0")
print("Expected: ValueError")
print("Actual:")
try:
    result = _rescale_imshow_rgb(darray, vmin=50.0, vmax=50.0, robust=False)
    print(f"No error raised! Result: {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")

print("\n" + "="*50 + "\n")

# Test case 3: For comparison - existing validation works (vmax=None, vmin too high)
print("Test 3: vmin=500.0, vmax=None (existing validation)")
print("Expected: ValueError")
print("Actual:")
try:
    result = _rescale_imshow_rgb(darray, vmin=500.0, vmax=None, robust=False)
    print(f"No error raised! Result: {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")
```

<details>

<summary>
Output demonstrates silent incorrect behavior when vmin >= vmax
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/xarray/plot/utils.py:778: RuntimeWarning: invalid value encountered in divide
  darray = ((darray.astype("f8") - vmin) / (vmax - vmin)).astype("f4")
Test 1: vmin=100.0, vmax=0.0
Expected: ValueError
Actual:
No error raised! Result: [[[0.5 0.5 0.5]]]

==================================================

Test 2: vmin=50.0, vmax=50.0
Expected: ValueError
Actual:
No error raised! Result: [[[nan nan nan]]]

==================================================

Test 3: vmin=500.0, vmax=None (existing validation)
Expected: ValueError
Actual:
ValueError raised: vmin=500.0 is less than the default vmax (1) - you must supply a vmax > vmin in this case.
```
</details>

## Why This Is A Bug

This violates the mathematical contract of linear rescaling and contradicts the function's own validation logic. The function uses the formula `(darray - vmin) / (vmax - vmin)` to rescale values from the interval [vmin, vmax] to [0, 1]. This operation fundamentally requires vmin < vmax because:

1. **Mathematical correctness**: When vmin > vmax, the denominator becomes negative, inverting the scaling direction and producing incorrect results where values above vmin become negative after scaling.

2. **Division by zero**: When vmin == vmax, the denominator is zero, causing NaN values to propagate through the output array.

3. **Inconsistent validation**: The function already validates this constraint in lines 762-766 (when vmax is None) and lines 769-773 (when vmin is None), establishing that vmin < vmax is a required invariant. The missing validation when both are provided is clearly an oversight.

4. **Code comment contradiction**: Line 759 explicitly states the intent to "check that an interval between them exists" but this check is missing for the both-provided case.

5. **Silent data corruption**: Instead of failing fast with a clear error, the function produces mathematically incorrect output that could lead to misinterpretation of scientific data visualizations.

## Relevant Context

The `_rescale_imshow_rgb` function is part of xarray's plotting utilities and is called by the public `imshow` API when displaying RGB(A) data. While the function is marked as private (underscore prefix), it directly impacts end users through the public API.

The existing code structure at lines 760-773 in `/home/npc/miniconda/lib/python3.13/site-packages/xarray/plot/utils.py` shows three conditional branches:
- `if vmax is None`: Sets default vmax and validates against provided vmin
- `elif vmin is None`: Sets default vmin (0) and validates against provided vmax
- `else`: **Missing validation** when both are provided

This is particularly problematic because users might accidentally swap vmin/vmax arguments or use them thinking they represent a range in either direction, leading to subtle visualization errors in scientific data analysis.

Documentation: https://docs.xarray.dev/en/stable/generated/xarray.plot.imshow.html

## Proposed Fix

```diff
--- a/xarray/plot/utils.py
+++ b/xarray/plot/utils.py
@@ -771,6 +771,11 @@ def _rescale_imshow_rgb(darray, vmin, vmax, robust):
                 f"vmax={vmax!r} is less than the default vmin (0) - you must supply "
                 "a vmin < vmax in this case."
             )
+    else:
+        # Both vmin and vmax are provided, validate they form a valid interval
+        if vmin >= vmax:
+            raise ValueError(
+                f"vmin ({vmin!r}) must be less than vmax ({vmax!r})."
+            )
     # Scale interval [vmin .. vmax] to [0 .. 1], with darray as 64-bit float
     # to avoid precision loss, integer over/underflow, etc with extreme inputs.
     # After scaling, downcast to 32-bit float.  This substantially reduces
```