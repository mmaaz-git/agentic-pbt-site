# Bug Report: pd.qcut Crashes with Denormal Floats Due to Precision Rounding Issue

**Target**: `pandas.qcut`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pd.qcut` crashes with a misleading ValueError about "missing values" when binning arrays containing denormal floating-point numbers, even though the input contains no missing values. The actual cause is `np.around` returning NaN when rounding denormal floats with extremely high precision digits.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st, assume
import pandas as pd


@given(
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=2, max_size=50),
    q=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=200)
def test_qcut_assigns_all_values(values, q):
    assume(len(set(values)) >= q)
    try:
        result = pd.qcut(values, q=q)
        assert len(result) == len(values)
        non_null = result.notna().sum()
        assert non_null == len(values)
    except ValueError as e:
        if "Bin edges must be unique" in str(e):
            assume(False)
        raise

# Run the test
if __name__ == "__main__":
    test_qcut_assigns_all_values()
```

<details>

<summary>
**Failing input**: `values=[0.0, 1.1125369292536007e-308], q=2`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 24, in <module>
    test_qcut_assigns_all_values()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 6, in test_qcut_assigns_all_values
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=2, max_size=50),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 13, in test_qcut_assigns_all_values
    result = pd.qcut(values, q=q)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py", line 340, in qcut
    fac, bins = _bins_to_cuts(
                ~~~~~~~~~~~~~^
        x_idx,
        ^^^^^^
    ...<4 lines>...
        duplicates=duplicates,
        ^^^^^^^^^^^^^^^^^^^^^^
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
Falsifying example: test_qcut_assigns_all_values(
    values=[0.0, 1.1125369292536007e-308],
    q=2,
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import traceback

values = [0.0, 2.458649946504791e-307]
q = 2

print("Input values:", values)
print("q:", q)
print("\nAttempting pd.qcut(values, q=q)...")
print("-" * 50)

try:
    result = pd.qcut(values, q=q)
    print("Result:", result)
except Exception as e:
    print("Exception:", type(e).__name__)
    print("Error message:", str(e))
    print("\nFull traceback:")
    traceback.print_exc()
```

<details>

<summary>
ValueError: missing values must be missing in the same location both left and right sides
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/repo.py", line 13, in <module>
    result = pd.qcut(values, q=q)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py", line 340, in qcut
    fac, bins = _bins_to_cuts(
                ~~~~~~~~~~~~~^
        x_idx,
        ^^^^^^
    ...<4 lines>...
        duplicates=duplicates,
        ^^^^^^^^^^^^^^^^^^^^^^
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
Input values: [0.0, 2.458649946504791e-307]
q: 2

Attempting pd.qcut(values, q=q)...
--------------------------------------------------
Exception: ValueError
Error message: missing values must be missing in the same location both left and right sides

Full traceback:
```
</details>

## Why This Is A Bug

The `pd.qcut` function should handle any valid numeric array according to its documentation which states it accepts "1d ndarray or Series" without mentioning restrictions on value ranges. The input `[0.0, 2.458649946504791e-307]` contains valid IEEE 754 denormal (subnormal) floating-point numbers.

The root cause is in the `_round_frac` function in `pandas/core/reshape/tile.py:615`. When processing denormal floats (values near 1e-307), it calculates `digits = 309` for precision rounding. However, `np.around(value, 309)` returns NaN for denormal floats, not the expected rounded value. This NaN then propagates through the bin formatting process, causing IntervalArray validation to fail with the misleading error about "missing values."

The error message provides no useful debugging information - users have no way to know the issue is related to numerical precision with extremely small values. This affects scientists and engineers working with legitimate small measurements (molecular concentrations, quantum physics constants).

## Relevant Context

The bug occurs in the following call chain:
1. `pd.qcut()` calculates quantile bins successfully
2. `_bins_to_cuts()` calls `_format_labels()` with default precision=3
3. `_format_labels()` calls `_round_frac()` for each bin edge
4. `_round_frac()` for denormal floats calculates digits > 308 (e.g., 309)
5. `np.around(denormal_value, 309)` returns NaN instead of a rounded value
6. IntervalIndex creation fails when breaks contain NaN values inconsistently

Documentation link: https://pandas.pydata.org/docs/reference/api/pandas.qcut.html

Related code locations:
- `pandas/core/reshape/tile.py:615` - _round_frac function
- `pandas/core/reshape/tile.py:565` - formatter using _round_frac
- `pandas/core/arrays/interval.py:664` - validation error

## Proposed Fix

The issue can be fixed by adding a check in `_round_frac` to handle cases where `np.around` would return NaN for extremely high precision digits with denormal floats:

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -621,10 +621,17 @@ def _round_frac(x, precision: int):
     else:
         frac, whole = np.modf(x)
         if whole == 0:
             digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
         else:
             digits = precision
-        return np.around(x, digits)
+
+        # np.around returns NaN for denormal floats with very high digits (>308)
+        # In such cases, return the original value or 0 based on magnitude
+        result = np.around(x, digits)
+        if np.isnan(result) and np.isfinite(x):
+            # For denormal floats that can't be rounded, return 0 if tiny enough
+            return 0.0 if abs(x) < 1e-300 else x
+        return result
```