# Bug Report: pandas.core.reshape.tile.cut Precision Overflow Causes All Data to Become NaN

**Target**: `pandas.core.reshape.tile.cut`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`pd.cut()` silently converts all valid numeric data to NaN when the data range is extremely small (< 1e-300), due to precision overflow in the internal `_round_frac` function that produces NaN bin edges.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import pandas as pd

def check_cut_preserves_non_na(data, bins):
    """Helper function to check the cut invariant"""
    result = pd.cut(data, bins=bins)

    non_na_input = sum(1 for x in data if not pd.isna(x))
    non_na_result = sum(1 for x in result if not pd.isna(x))

    assert non_na_input == non_na_result, f"Expected {non_na_input} non-NA values but got {non_na_result}"

@given(
    data=st.lists(
        st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=10,
        max_size=100,
    ),
    bins=st.integers(min_value=2, max_value=10),
)
def test_cut_all_values_assigned_to_bins(data, bins):
    assume(len(set(data)) > 1)
    check_cut_preserves_non_na(data, bins)
```

<details>

<summary>
**Failing input**: `data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.225073858507203e-309], bins=2`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
Testing with the specific failing input...
Test failed as expected: Expected 10 non-NA values but got 0

Running Hypothesis to find more failures...
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 43, in <module>
  |     test_cut_all_values_assigned_to_bins()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 14, in test_cut_all_values_assigned_to_bins
  |     data=st.lists(
  |                ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 28, in test_cut_all_values_assigned_to_bins
    |     check_cut_preserves_non_na(data, bins)
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
    |   File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 6, in check_cut_preserves_non_na
    |     result = pd.cut(data, bins=bins)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py", line 257, in cut
    |     fac, bins = _bins_to_cuts(
    |                 ~~~~~~~~~~~~~^
    |         x_idx,
    |         ^^^^^^
    |     ...<6 lines>...
    |         ordered=ordered,
    |         ^^^^^^^^^^^^^^^^
    |     )
    |     ^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py", line 483, in _bins_to_cuts
    |     labels = _format_labels(
    |         bins, precision, right=right, include_lowest=include_lowest
    |     )
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py", line 577, in _format_labels
    |     return IntervalIndex.from_breaks(breaks, closed=closed)
    |            ~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/interval.py", line 275, in from_breaks
    |     array = IntervalArray.from_breaks(
    |         breaks, closed=closed, copy=copy, dtype=dtype
    |     )
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/interval.py", line 464, in from_breaks
    |     return cls.from_arrays(breaks[:-1], breaks[1:], closed, copy=copy, dtype=dtype)
    |            ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/interval.py", line 552, in from_arrays
    |     cls._validate(left, right, dtype=dtype)
    |     ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/interval.py", line 664, in _validate
    |     raise ValueError(msg)
    | ValueError: missing values must be missing in the same location both left and right sides
    | Falsifying example: test_cut_all_values_assigned_to_bins(
    |     data=[0.0,
    |      0.0,
    |      0.0,
    |      0.0,
    |      0.0,
    |      0.0,
    |      0.0,
    |      0.0,
    |      0.0,
    |      -2.225073858507203e-309],
    |     bins=2,  # or any other generated value
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/interval.py:661
    |         /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py:638
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 28, in test_cut_all_values_assigned_to_bins
    |     check_cut_preserves_non_na(data, bins)
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
    |   File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 11, in check_cut_preserves_non_na
    |     assert non_na_input == non_na_result, f"Expected {non_na_input} non-NA values but got {non_na_result}"
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Expected 10 non-NA values but got 0
    | Falsifying example: test_cut_all_values_assigned_to_bins(
    |     data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.225073858507203e-309],
    |     bins=2,  # or any other generated value
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_dtype.py:329
    |         /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_internal.py:939
    |         /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/algorithms.py:133
    |         /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/algorithms.py:202
    |         /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/algorithms.py:297
    |         (and 45 more with settings.verbosity >= verbose)
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

# The minimal failing input from the bug report
data = [0.0] * 9 + [2.225073858507e-311]
print(f"Input data: {data}")
print(f"Number of input values: {len(data)}")
print(f"Data range: min={min(data)}, max={max(data)}")
print(f"Data range size: {max(data) - min(data)}")
print()

# Call pd.cut with 2 bins and get the bins back
result, bins = pd.cut(data, bins=2, retbins=True)

print(f"Bins returned: {bins}")
print(f"Result categories: {result.categories}")
print(f"Number of categories: {len(result.categories)}")
print()

# Check how many values became NaN
print(f"Input data values that are NaN: {sum(1 for x in data if pd.isna(x))}")
print(f"Result values that are NaN: {sum(1 for x in result if pd.isna(x))}")
print()

# Print the actual result values
print("Result values:")
for i, val in enumerate(result):
    print(f"  data[{i}] = {data[i]} -> {val}")
```

<details>

<summary>
RuntimeWarning and all values become NaN
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
Input data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.225073858507e-311]
Number of input values: 10
Data range: min=0.0, max=2.225073858507e-311
Data range size: 2.225073858507e-311

Bins returned: [-2.22507386e-314  1.11253693e-311  2.22507386e-311]
Result categories: IntervalIndex([], dtype='interval[float64, right]')
Number of categories: 0

Input data values that are NaN: 0
Result values that are NaN: 10

Result values:
  data[0] = 0.0 -> nan
  data[1] = 0.0 -> nan
  data[2] = 0.0 -> nan
  data[3] = 0.0 -> nan
  data[4] = 0.0 -> nan
  data[5] = 0.0 -> nan
  data[6] = 0.0 -> nan
  data[7] = 0.0 -> nan
  data[8] = 0.0 -> nan
  data[9] = 2.225073858507e-311 -> nan
```
</details>

## Why This Is A Bug

This violates `pd.cut()`'s fundamental contract that it should bin all valid numeric input data. According to the pandas documentation:

1. **From the docstring**: "Bin values into discrete intervals" - the function should create bins that contain the input values
2. **Expected behavior**: All 10 valid numeric values should be assigned to one of the 2 requested bins
3. **Actual behavior**: All values become NaN despite being valid floats, and 0 categories are created instead of 2

The bug occurs when data has an extremely small range (near machine epsilon, < 1e-300). The internal `_round_frac` function at line 624 of `/pandas/core/reshape/tile.py` calculates an excessively large `digits` parameter:

```python
digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
```

For `frac = 2.225e-311`, this results in `digits ≈ 316`, which exceeds float64's precision limit of ~15-17 decimal digits. When `np.around(x, 316)` is called, it overflows and returns NaN for all bin edges. These NaN bin edges then create an empty IntervalIndex with no valid intervals, causing all input values to map to NaN.

## Relevant Context

- The bug affects scientific computing applications that work with very small floating-point values
- Hypothesis found multiple similar failures with values around 1e-309 to 1e-311
- The issue occurs in both positive and negative small value ranges
- NumPy's `np.around()` function has a practical limit on the `decimals` parameter that should be respected
- Documentation: https://pandas.pydata.org/docs/reference/api/pandas.cut.html

## Proposed Fix

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -621,6 +621,9 @@ def _round_frac(x, precision: int):
     else:
         frac, whole = np.modf(x)
         if whole == 0:
             digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
+            # Clamp digits to prevent np.around overflow
+            # float64 has ~15-17 decimal digits of precision
+            digits = min(digits, 15)
         else:
             digits = precision
         return np.around(x, digits)
```