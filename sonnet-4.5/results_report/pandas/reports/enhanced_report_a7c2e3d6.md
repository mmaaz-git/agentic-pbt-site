# Bug Report: pandas.core.reshape.tile.cut - Crash with Extremely Small Float Values

**Target**: `pandas.core.reshape.tile.cut`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `pd.cut()` function crashes with a confusing "missing values" error when binning valid but extremely small floating-point values near the limits of float64 representation, due to an internal numerical precision overflow.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings, assume


@given(
    st.lists(st.floats(min_value=-1e-300, max_value=1e-300, allow_nan=False, allow_infinity=False), min_size=2, max_size=10),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=100)
def test_cut_handles_tiny_floats_without_crash(values, n_bins):
    assume(len(set(values)) > 1)

    x = pd.Series(values)
    try:
        result = pd.cut(x, bins=n_bins)
    except ValueError as e:
        if "missing values must be missing" in str(e):
            raise AssertionError(f"cut() crashed on valid input: {values}") from e
        raise

if __name__ == "__main__":
    test_cut_handles_tiny_floats_without_crash()
```

<details>

<summary>
**Failing input**: `values=[0.0, -1.1125369292536007e-308], n_bins=2`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 15, in test_cut_handles_tiny_floats_without_crash
    result = pd.cut(x, bins=n_bins)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py", line 257, in cut
    fac, bins = _bins_to_cuts(
                ~~~~~~~~~~~~~^
        x_idx,
        ^^^^^^
    ...<6 lines>...
        ordered=ordered,
        ^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py", line 483, in _bins_to_cuts
    labels = _format_labels(
        bins, precision, right=right, include_lowest=include_lowest
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py", line 577, in _format_labels
    return IntervalIndex.from_breaks(breaks, closed=closed)
           ~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/interval.py", line 275, in from_breaks
    array = IntervalArray.from_breaks(
        breaks, closed=closed, copy=copy, dtype=dtype
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/interval.py", line 464, in from_breaks
    return cls.from_arrays(breaks[:-1], breaks[1:], closed, copy=copy, dtype=dtype)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/interval.py", line 552, in from_arrays
    cls._validate(left, right, dtype=dtype)
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/interval.py", line 664, in _validate
    raise ValueError(msg)
ValueError: missing values must be missing in the same location both left and right sides

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 22, in <module>
    test_cut_handles_tiny_floats_without_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 6, in test_cut_handles_tiny_floats_without_crash
    st.lists(st.floats(min_value=-1e-300, max_value=1e-300, allow_nan=False, allow_infinity=False), min_size=2, max_size=10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 18, in test_cut_handles_tiny_floats_without_crash
    raise AssertionError(f"cut() crashed on valid input: {values}") from e
AssertionError: cut() crashed on valid input: [0.0, -1.1125369292536007e-308]
Falsifying example: test_cut_handles_tiny_floats_without_crash(
    values=[0.0, -1.1125369292536007e-308],
    n_bins=2,
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

values = [1.1125369292536007e-308, -6.312184867281418e-301]
x = pd.Series(values)

print(f"Input values: {x.tolist()}")
print(f"All values are valid (non-NaN): {x.notna().all()}")
print(f"Data range: [{x.min()}, {x.max()}]")
print()

try:
    result = pd.cut(x, bins=2)
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")
```

<details>

<summary>
ValueError with misleading "missing values" message despite valid input
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
Input values: [1.1125369292536007e-308, -6.312184867281418e-301]
All values are valid (non-NaN): True
Data range: [-6.312184867281418e-301, 1.1125369292536007e-308]

ValueError raised: missing values must be missing in the same location both left and right sides
```
</details>

## Why This Is A Bug

This violates expected behavior because `pd.cut()` crashes on valid float64 input with a misleading error message. The inputs are legitimate floating-point values within the valid float64 range (±2.225e-308 to ±1.798e+308), contain no NaN or infinity values, yet trigger an internal validation error about "missing values" that don't exist in the input.

The root cause is in the `_round_frac()` function at `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py:615`. When formatting bin labels for extremely small values like 1.1125369292536007e-308, it calculates a rounding precision of 310+ decimal places. NumPy's `np.around()` cannot handle such extreme precision and returns NaN, which then propagates through the IntervalIndex creation, ultimately triggering the confusing "missing values must be missing in the same location" error.

The pandas documentation for `cut()` does not mention any restrictions on the magnitude of input values, and users would reasonably expect the function to either successfully bin these valid floats or provide a clear error about numerical precision limitations.

## Relevant Context

The issue occurs specifically when:
1. Input contains extremely small float values (near 1e-308, the lower limit of normalized float64)
2. These values have mixed signs or include zero
3. The `_format_labels()` function tries to format bin edges for display

The problematic code path:
- `pd.cut()` → `_bins_to_cuts()` → `_format_labels()` → `_round_frac()`
- In `_round_frac()`, for tiny values where `whole == 0`, it computes: `digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision`
- For `frac = 1.1125369292536007e-308`, this results in `digits = 310`
- `np.around(1.1125369292536007e-308, 310)` returns NaN
- The NaN breaks interval creation with an unrelated error message

Documentation: https://pandas.pydata.org/docs/reference/api/pandas.cut.html
Source code: pandas/core/reshape/tile.py

## Proposed Fix

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -615,6 +615,10 @@ def _round_frac(x, precision: int):
         frac, whole = np.modf(x)
         if whole == 0:
             digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
+            # Limit digits to avoid numerical precision issues with np.around
+            # np.around fails for extremely large decimal places (>308)
+            if digits > 308:
+                return x  # Return unchanged for values requiring extreme precision
         else:
             digits = precision
         return np.around(x, digits)
```