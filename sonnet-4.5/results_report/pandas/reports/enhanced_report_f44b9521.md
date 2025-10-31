# Bug Report: pandas.core.reshape.tile.cut - Silent Data Loss with Tiny Float Values

**Target**: `pandas.core.reshape.tile.cut`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `pd.cut()` function silently loses all data (returns all NaN values) when binning very small positive float values near machine epsilon due to np.around() returning NaN for extreme precision values.

## Property-Based Test

```python
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume


@given(
    st.lists(st.floats(min_value=0, max_value=1e-300, allow_nan=False, allow_infinity=False), min_size=2, max_size=10),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=100)
def test_cut_preserves_data_with_tiny_floats(values, n_bins):
    assume(len(set(values)) > 1)
    assume(all(v >= 0 for v in values))

    x = pd.Series(values)
    result = pd.cut(x, bins=n_bins)

    assert result.notna().sum() == x.notna().sum(), \
        f"Data loss: {x.notna().sum()} valid inputs became {result.notna().sum()} valid outputs"


if __name__ == "__main__":
    test_cut_preserves_data_with_tiny_floats()
```

<details>

<summary>
**Failing input**: `values=[0.0, 1.1125369292536007e-308], n_bins=2`
</summary>
```
Trying example: test_cut_preserves_data_with_tiny_floats(
    values=[0.0, 0.0],
    n_bins=2,
)
Trying example: test_cut_preserves_data_with_tiny_floats(
    values=[5e-324, 0.0],
    n_bins=2,
)
Traceback (most recent call last):
  File "<string>", line 16, in test_cut_preserves_data_with_tiny_floats
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
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py", line 443, in _bins_to_cuts
    raise ValueError(
    ...<2 lines>...
    )
ValueError: Bin edges must be unique: Index([0.0, 0.0, 5e-324], dtype='float64').
You can drop duplicate edges by setting the 'duplicates' kwarg

Trying example: test_cut_preserves_data_with_tiny_floats(
    values=[5e-324, 8.855575976502743e-301, 5.73902817260794e-301],
    n_bins=3,
)
Trying example: test_cut_preserves_data_with_tiny_floats(
    values=[8.855575976502743e-301,
     8.855575976502743e-301,
     5.73902817260794e-301],
    n_bins=3,
)
Trying example: test_cut_preserves_data_with_tiny_floats(
    values=[5.73902817260794e-301,
     8.855575976502743e-301,
     5.73902817260794e-301],
    n_bins=3,
)
Trying example: test_cut_preserves_data_with_tiny_floats(
    values=[5.73902817260794e-301,
     5.73902817260794e-301,
     5.73902817260794e-301],
    n_bins=3,
)
Trying example: test_cut_preserves_data_with_tiny_floats(
    values=[5.73902817260794e-301,
     8.855575976502743e-301,
     8.855575976502743e-301],
    n_bins=3,
)
Trying example: test_cut_preserves_data_with_tiny_floats(
    values=[8.855575976502743e-301,
     8.855575976502743e-301,
     8.855575976502743e-301],
    n_bins=3,
)
Trying example: test_cut_preserves_data_with_tiny_floats(
    values=[5.73902817260794e-301,
     5.73902817260794e-301,
     8.855575976502743e-301],
    n_bins=3,
)
Trying example: test_cut_preserves_data_with_tiny_floats(
    values=[2.9138462228746394e-301,
     2.975956096493247e-301,
     2.0220561606099424e-301,
     2.837979954706525e-301,
     6.449351374694269e-302],
    n_bins=2,
)
Trying example: test_cut_preserves_data_with_tiny_floats(
    values=[5e-324, 0.0],
    n_bins=2,
)
Traceback (most recent call last):
  File "<string>", line 16, in test_cut_preserves_data_with_tiny_floats
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
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py", line 443, in _bins_to_cuts
    raise ValueError(
    ...<2 lines>...
    )
ValueError: Bin edges must be unique: Index([0.0, 0.0, 5e-324], dtype='float64').
You can drop duplicate edges by setting the 'duplicates' kwarg

Trying example: test_cut_preserves_data_with_tiny_floats(
    values=[0.0, 5e-324],
    n_bins=2,
)
Traceback (most recent call last):
  File "<string>", line 16, in test_cut_preserves_data_with_tiny_floats
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
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py", line 443, in _bins_to_cuts
    raise ValueError(
    ...<2 lines>...
    )
ValueError: Bin edges must be unique: Index([0.0, 0.0, 5e-324], dtype='float64').
You can drop duplicate edges by setting the 'duplicates' kwarg

Trying example: test_cut_preserves_data_with_tiny_floats(
    values=[0.0, 5e-324],
    n_bins=5,
)
Traceback (most recent call last):
  File "<string>", line 16, in test_cut_preserves_data_with_tiny_floats
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
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py", line 443, in _bins_to_cuts
    raise ValueError(
    ...<2 lines>...
    )
ValueError: Bin edges must be unique: Index([0.0, 0.0, 0.0, 5e-324, 5e-324, 5e-324], dtype='float64').
You can drop duplicate edges by setting the 'duplicates' kwarg

Trying example: test_cut_preserves_data_with_tiny_floats(
    values=[0.0, 5e-324],
    n_bins=4,
)
Traceback (most recent call last):
  File "<string>", line 16, in test_cut_preserves_data_with_tiny_floats
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
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py", line 443, in _bins_to_cuts
    raise ValueError(
    ...<2 lines>...
    )
ValueError: Bin edges must be unique: Index([0.0, 0.0, 0.0, 5e-324, 5e-324], dtype='float64').
You can drop duplicate edges by setting the 'duplicates' kwarg

Trying example: test_cut_preserves_data_with_tiny_floats(
    values=[0.0, 5e-324],
    n_bins=3,
)
Traceback (most recent call last):
  File "<string>", line 16, in test_cut_preserves_data_with_tiny_floats
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
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py", line 443, in _bins_to_cuts
    raise ValueError(
    ...<2 lines>...
    )
ValueError: Bin edges must be unique: Index([0.0, 0.0, 5e-324, 5e-324], dtype='float64').
You can drop duplicate edges by setting the 'duplicates' kwarg

Traceback (most recent call last):
  File "<string>", line 21, in <module>
    test_cut_preserves_data_with_tiny_floats()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "<string>", line 7, in test_cut_preserves_data_with_tiny_floats
    st.lists(st.floats(min_value=0, max_value=1e-300, allow_nan=False, allow_infinity=False), min_size=2, max_size=10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "<string>", line 16, in test_cut_preserves_data_with_tiny_floats
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
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py", line 443, in _bins_to_cuts
    raise ValueError(
    ...<2 lines>...
    )
ValueError: Bin edges must be unique: Index([0.0, 0.0, 5e-324], dtype='float64').
You can drop duplicate edges by setting the 'duplicates' kwarg
Falsifying example: test_cut_preserves_data_with_tiny_floats(
    values=[0.0, 5e-324],
    n_bins=2,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/function_base.py:156
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/_config/config.py:138
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/_config/config.py:659
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/_config/config.py:685
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/base.py:1346
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/base.py:1354
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/base.py:1371
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/base.py:1373
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/base.py:1377
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/indexes/base.py:1379
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py:442
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/console.py:18
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/console.py:30
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/printing.py:279
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/printing.py:335
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/printing.py:344
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/printing.py:349
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/printing.py:569
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import numpy as np

values = [0.0, 1.1125369292536007e-308]
x = pd.Series(values)

print(f"Input values: {x.tolist()}")
print(f"Input non-null count: {x.notna().sum()}")

result = pd.cut(x, bins=2)

print(f"Result: {result.tolist()}")
print(f"Result non-null count: {result.notna().sum()}")
print(f"Expected: 2 non-null values, Got: {result.notna().sum()}")

result, bins = pd.cut(x, bins=2, retbins=True)
print(f"\nBins computed: {bins}")
print(f"Result categories: {result.cat.categories}")

# Additional debugging info
print(f"\nDebug: x.min() = {x.min()}")
print(f"Debug: x.max() = {x.max()}")
print(f"Debug: Range = {x.max() - x.min()}")
```

<details>

<summary>
Silent data loss with tiny float values
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:46: RuntimeWarning: invalid value encountered in divide
  result = getattr(arr, method)(*args, **kwds)
Input values: [0.0, 1.1125369292536007e-308]
Input non-null count: 2
Result: [nan, nan]
Result non-null count: 0
Expected: 2 non-null values, Got: 0

Bins computed: [-1.11253693e-311  5.56268465e-309  1.11253693e-308]
Result categories: IntervalIndex([], dtype='interval[float64, right]')

Debug: x.min() = 0.0
Debug: x.max() = 1.1125369292536007e-308
Debug: Range = 1.1125369292536007e-308
```
</details>

## Why This Is A Bug

This violates the fundamental contract of `pd.cut()`: to bin all valid (non-NaN) input values into discrete intervals. The function successfully computes the bin edges (as shown by `retbins=True`), but fails during the IntervalIndex creation step, resulting in complete data loss.

The root cause is in the `_round_frac()` function at pandas/core/reshape/tile.py:615-627. When processing very small float values (near machine epsilon ~2.2e-308), this function calculates an extremely large precision value (310+ digits) for rounding. However, NumPy's `np.around()` function returns NaN when called with precision values >= 310 for such small floats. This causes all bin edges to become NaN during the formatting step, which results in an empty IntervalIndex and all output values becoming NaN.

This is a **high-severity** bug because:
1. It causes **complete silent data corruption** - all valid data becomes NaN with only a warning about division
2. Scientific computing routinely uses values near machine epsilon (quantum mechanics, particle physics, molecular simulations)
3. The failure is non-obvious - bins are computed correctly but the formatting step fails
4. Users have no workaround except manually creating bins

## Relevant Context

The issue manifests in two ways depending on the exact values:
1. For values like [0.0, 1.1125e-308], it silently returns all NaN (data loss)
2. For even smaller values like [0.0, 5e-324], it raises a ValueError about duplicate bin edges

Both failures stem from the same root cause: `np.around()` returning NaN or rounding to zero when precision exceeds practical limits. The function correctly identifies that tiny values need high precision but doesn't account for NumPy's precision limitations.

Documentation reference: https://pandas.pydata.org/docs/reference/api/pandas.cut.html
Source code location: /pandas/core/reshape/tile.py lines 615-627

## Proposed Fix

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -622,8 +622,11 @@ def _round_frac(x, precision: int):
         frac, whole = np.modf(x)
         if whole == 0:
             digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
+            # Limit precision to avoid np.around() returning NaN for tiny floats
+            # np.around() returns NaN when precision >= ~310 for values near machine epsilon
+            digits = min(digits, 308)
         else:
             digits = precision
         return np.around(x, digits)
```