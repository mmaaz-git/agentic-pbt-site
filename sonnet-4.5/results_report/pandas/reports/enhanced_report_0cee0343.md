# Bug Report: pandas.cut duplicates='drop' Crashes with Subnormal Float Values

**Target**: `pandas.cut()` and `pandas.qcut()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pd.cut()` with `duplicates='drop'` crashes with a cryptic error when processing data containing subnormal float values (near machine epsilon), even though the error message explicitly suggests using `duplicates='drop'` to resolve duplicate bin edges.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test that discovered the pandas.cut bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from hypothesis import given, strategies as st, settings, example

@given(
    x=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=5, max_size=50),
    bins=st.integers(min_value=2, max_value=10)
)
@example(x=[0.0, 0.0, 0.0, 0.0, 5e-324], bins=2)  # Known failing example
@settings(max_examples=300, deadline=None)
def test_cut_preserves_length(x, bins):
    """Test that pd.cut with duplicates='drop' preserves array length."""
    result = pd.cut(x, bins=bins, duplicates='drop')
    assert len(result) == len(x), f"Result length {len(result)} != input length {len(x)}"

if __name__ == "__main__":
    # Run the test
    print("Running property-based test for pandas.cut...")
    print("This test checks that pd.cut(x, bins, duplicates='drop') preserves input length.")
    print()

    try:
        test_cut_preserves_length()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed with assertion: {e}")
    except Exception as e:
        print(f"Test failed with exception: {type(e).__name__}: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
```

<details>

<summary>
**Failing input**: `x=[0.0, 0.0, 0.0, 0.0, 5e-324], bins=2`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:57: RuntimeWarning: invalid value encountered in divide
  return bound(*args, **kwds)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 28, in <module>
    test_cut_preserves_length()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 11, in test_cut_preserves_length
    x=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=5, max_size=50),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 18, in test_cut_preserves_length
    result = pd.cut(x, bins=bins, duplicates='drop')
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
Falsifying explicit example: test_cut_preserves_length(
    x=[0.0, 0.0, 0.0, 0.0, 5e-324],
    bins=2,
)
Running property-based test for pandas.cut...
This test checks that pd.cut(x, bins, duplicates='drop') preserves input length.

Test failed with exception: ValueError: missing values must be missing in the same location both left and right sides

Full traceback:
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of pandas.cut duplicates='drop' bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd

# Test case from the bug report
x = [0.0, 0.0, 0.0, 0.0, 5e-324]

print("=" * 60)
print("Test Case 1: pd.cut with bins=2")
print("Input: x =", x)
print()

# First try without duplicates='drop'
print("Attempting pd.cut(x, bins=2):")
try:
    result = pd.cut(x, bins=2)
    print("Success! Result:", result)
except ValueError as e:
    print(f"Error: {e}")
    print()

    # Following the error message's advice
    print("Following error message advice - using duplicates='drop':")
    print("Attempting pd.cut(x, bins=2, duplicates='drop'):")
    try:
        result = pd.cut(x, bins=2, duplicates='drop')
        print("Success! Result:", result)
    except ValueError as e:
        print(f"Error: {e}")

print()
print("=" * 60)
print("Test Case 2: pd.qcut with q=2")
x2 = [0.0]*9 + [2.225073858507e-311]
print("Input: x =", x2)
print()

print("Attempting pd.qcut(x, q=2, duplicates='drop'):")
try:
    result = pd.qcut(x2, q=2, duplicates='drop')
    print("Success! Result:", result)
except ValueError as e:
    print(f"Error: {e}")

print()
print("=" * 60)
print("Test Case 3: Works with larger values")
x3 = [0.0, 0.0, 0.0, 0.0, 0.1]
print("Input: x =", x3)
print()

print("Attempting pd.cut(x, bins=2, duplicates='drop'):")
try:
    result = pd.cut(x3, bins=2, duplicates='drop')
    print("Success! Result:", result.tolist())
except ValueError as e:
    print(f"Error: {e}")
```

<details>

<summary>
ValueError when following error message advice to use duplicates='drop'
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:57: RuntimeWarning: invalid value encountered in divide
  return bound(*args, **kwds)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:57: RuntimeWarning: invalid value encountered in divide
  return bound(*args, **kwds)
============================================================
Test Case 1: pd.cut with bins=2
Input: x = [0.0, 0.0, 0.0, 0.0, 5e-324]

Attempting pd.cut(x, bins=2):
Error: Bin edges must be unique: Index([0.0, 0.0, 5e-324], dtype='float64').
You can drop duplicate edges by setting the 'duplicates' kwarg

Following error message advice - using duplicates='drop':
Attempting pd.cut(x, bins=2, duplicates='drop'):
Error: missing values must be missing in the same location both left and right sides

============================================================
Test Case 2: pd.qcut with q=2
Input: x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.225073858507e-311]

Attempting pd.qcut(x, q=2, duplicates='drop'):
Error: missing values must be missing in the same location both left and right sides

============================================================
Test Case 3: Works with larger values
Input: x = [0.0, 0.0, 0.0, 0.0, 0.1]

Attempting pd.cut(x, bins=2, duplicates='drop'):
Success! Result: [Interval(-0.0001, 0.05, closed='right'), Interval(-0.0001, 0.05, closed='right'), Interval(-0.0001, 0.05, closed='right'), Interval(-0.0001, 0.05, closed='right'), Interval(0.05, 0.1, closed='right')]
```
</details>

## Why This Is A Bug

The `duplicates='drop'` parameter is documented to "remove non-unique bin edges" and allow the function to proceed. The initial error message explicitly instructs users: "You can drop duplicate edges by setting the 'duplicates' kwarg". However, following this advice causes a crash with an unrelated error about "missing values must be missing in the same location".

The root cause is in the `_round_frac()` function in `/pandas/core/reshape/tile.py`. When formatting bin labels for extremely small values like `5e-324`:

1. The function calculates `digits = -int(np.floor(np.log10(5e-324))) - 1 + precision = 326`
2. NumPy's `np.around()` returns `NaN` when digits >= 310 (undocumented NumPy limitation)
3. This causes the formatted breaks to contain `[0.0, nan]` instead of `[0.0, 5e-324]`
4. `IntervalIndex.from_breaks()` fails because it cannot create valid intervals with NaN values

This violates the documented contract that `duplicates='drop'` should handle duplicate edges gracefully. The error message actively misleads users into triggering the bug. While subnormal floats (5e-324 is the smallest positive float64) are edge cases, they are valid float64 values used in scientific computing, numerical analysis, and machine learning applications.

## Relevant Context

- **Subnormal floats**: Values like `5e-324` and `2.225073858507e-311` are subnormal (denormalized) floating-point numbers, which are valid IEEE 754 float64 values representing extremely small positive numbers near zero.
- **NumPy limitation**: `np.around()` silently returns NaN for `decimals` >= 310, which is not documented in NumPy's official documentation.
- **Pandas documentation**: The official pandas documentation for `pd.cut()` states that `duplicates='drop'` should "remove non-unique bin edges" with no mentioned limitations for small values.
- **Source location**: The bug is in `/pandas/core/reshape/tile.py` at line 627 in the `_round_frac()` function.

## Proposed Fix

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -615,14 +615,21 @@ def _round_frac(x, precision: int):
     """
     Round the fractional part of the given number
     """
     if not np.isfinite(x) or x == 0:
         return x
     else:
         frac, whole = np.modf(x)
         if whole == 0:
             digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
+            # NumPy's around() returns NaN for digits >= 310
+            # Clamp to a safe maximum to avoid NaN results
+            if digits >= 310:
+                # Return the original value for extremely small numbers
+                # that would cause np.around to return NaN
+                return x
         else:
             digits = precision
         return np.around(x, digits)
```