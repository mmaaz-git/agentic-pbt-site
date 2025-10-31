# Bug Report: pandas.cut ValueError with Denormal Float Values

**Target**: `pandas.cut`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pandas.cut` crashes with a misleading ValueError about "missing values" when attempting to bin data containing denormal (subnormal) floating-point numbers, even though no missing values are present in the input.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st, assume
import pandas as pd


@given(
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=2, max_size=50),
    bins=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=200)
def test_cut_assigns_all_values(values, bins):
    assume(len(set(values)) > 1)
    result = pd.cut(values, bins=bins)
    assert len(result) == len(values)

if __name__ == "__main__":
    test_cut_assigns_all_values()
```

<details>

<summary>
**Failing input**: `values=[0.0, -2.225073858507e-311], bins=2`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 16, in <module>
    test_cut_assigns_all_values()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 6, in test_cut_assigns_all_values
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=2, max_size=50),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 12, in test_cut_assigns_all_values
    result = pd.cut(values, bins=bins)
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
Falsifying example: test_cut_assigns_all_values(
    values=[0.0, -2.225073858507e-311],
    bins=2,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/interval.py:661
```
</details>

## Reproducing the Bug

```python
import pandas as pd

values = [2.2250738585e-313, -1.0]
result = pd.cut(values, bins=2)
print(result)
```

<details>

<summary>
ValueError: missing values must be missing in the same location both left and right sides
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/45/repo.py", line 4, in <module>
    result = pd.cut(values, bins=2)
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
```
</details>

## Why This Is A Bug

`pandas.cut` should handle all valid numeric input as specified in its documentation, which states that the `x` parameter accepts "array-like" input that is 1-dimensional. The input `[2.2250738585e-313, -1.0]` meets all documented requirements:

1. **Valid IEEE 754 numbers**: Both values are valid floating-point numbers. The value `2.2250738585e-313` is a denormal (subnormal) float, which is part of the IEEE 754 standard for representing very small numbers near zero.

2. **No missing values**: The error message claims "missing values must be missing in the same location" but the input contains no NaN or missing values.

3. **Documentation compliance**: The function's documentation does not exclude denormal floats or specify any precision limitations. It only requires the input to be array-like and 1-dimensional.

4. **Misleading error**: The RuntimeWarning about "invalid value encountered in divide" indicates that NaN values are being generated during internal calculations when creating bin edges, likely due to numerical precision issues with the extreme range difference between -1.0 and a denormal value near zero. This internal calculation failure then triggers the validation error about missing values.

## Relevant Context

**Denormal (Subnormal) Floats**: Values smaller than `2.2250738585072014e-308` (the smallest normal float64) but greater than zero are denormal floats. They provide gradual underflow in IEEE 754 floating-point arithmetic and are valid numeric values that should be handled correctly by numerical libraries.

**Error Location**: The error occurs in `/pandas/core/arrays/interval.py:664` within the `_validate` method when checking if missing values (NaN) are consistent between left and right interval boundaries. The NaN values appear to be generated inconsistently during bin edge calculations due to the extreme value range.

**Numerical Instability**: When calculating bins for the range `[-1.0, 2.2250738585e-313]`, the range calculation effectively becomes 1.0 (since the denormal is so close to zero). Operations on this range with adjustments (like the 0.1% range adjustment in line 404 of `tile.py`) can produce numerical instabilities.

**Related Code Paths**:
- `/pandas/core/reshape/tile.py:403-408`: Creates bins using `np.linspace` and applies range adjustments
- `/pandas/core/arrays/interval.py:657-664`: Validates that missing values are consistent between interval boundaries

## Proposed Fix

The issue stems from numerical instability when working with extreme value ranges. A defensive fix would detect and handle these cases explicitly:

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -398,6 +398,14 @@ def _get_bins(x_idx: Index, nbins: int, right: bool, precision: int, unit=None):
         mn, mx = (x_idx.min(), x_idx.max())
+
+    # Check for numerical stability issues with extreme ranges
+    if mn != mx:
+        value_range = mx - mn
+        if abs(value_range) < 1e-300 or abs(mn) < 1e-300 or abs(mx) < 1e-300:
+            raise ValueError(
+                f"Cannot create bins for values involving denormal floats or extreme ranges. "
+                f"Range: [{mn}, {mx}]. Consider scaling your data or using explicit bin edges.")
+
     if is_numeric_dtype(x_idx.dtype) and (np.isinf(mn) or np.isinf(mx)):
         # GH#24314
         raise ValueError(
```