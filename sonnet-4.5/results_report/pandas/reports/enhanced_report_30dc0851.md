# Bug Report: pandas.core.reshape.tile.cut Precision Rounding Excludes Values from Their Assigned Bins

**Target**: `pandas.core.reshape.tile.cut`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `pd.cut()` function with default `precision=3` parameter rounds bin boundaries for display, but these rounded boundaries are used for interval membership testing, causing values to be assigned to bins that mathematically don't contain them.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
import pandas as pd

@settings(max_examples=500)
@given(
    values=st.lists(
        st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100
    ),
    n_bins=st.integers(min_value=2, max_value=20)
)
def test_cut_respects_bin_boundaries(values, n_bins):
    assume(len(set(values)) >= 2)

    result = pd.cut(values, bins=n_bins)

    for i, (val, cat) in enumerate(zip(values, result)):
        if pd.notna(cat):
            left, right = cat.left, cat.right
            assert left < val <= right, \
                f"Value {val} at index {i} not in its assigned bin {cat}"

if __name__ == "__main__":
    test_cut_respects_bin_boundaries()
```

<details>

<summary>
**Failing input**: `values=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.15625], n_bins=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 25, in <module>
    test_cut_respects_bin_boundaries()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 5, in test_cut_respects_bin_boundaries
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 21, in test_cut_respects_bin_boundaries
    assert left < val <= right, \
           ^^^^^^^^^^^^^^^^^^^
AssertionError: Value 1.15625 at index 9 not in its assigned bin (0.578, 1.156]
Falsifying example: test_cut_respects_bin_boundaries(
    values=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.15625],
    n_bins=2,  # or any other generated value
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/36/hypo.py:22
```
</details>

## Reproducing the Bug

```python
import pandas as pd

values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.15625]
result = pd.cut(values, bins=2, precision=3)

val = 1.15625
cat = result[9]

print(f"Value: {val}")
print(f"Assigned interval: {cat}")
print(f"Value in interval? {val in cat}")
print()
print("Details:")
print(f"  Interval left boundary: {cat.left}")
print(f"  Interval right boundary: {cat.right}")
print(f"  Is value > right boundary? {val > cat.right}")
```

<details>

<summary>
Value assigned to interval that doesn't contain it
</summary>
```
Value: 1.15625
Assigned interval: (0.578, 1.156]
Value in interval? False

Details:
  Interval left boundary: 0.578
  Interval right boundary: 1.156
  Is value > right boundary? True
```
</details>

## Why This Is A Bug

This bug violates the fundamental contract of the `cut()` function: values should be contained within their assigned bins. The documentation states that `cut()` "Bin values into discrete intervals" and that the function segments data into bins based on specified boundaries.

The bug occurs due to the interaction between the range extension and precision rounding:

1. When `bins` is an integer, pandas extends the range by 0.1% on the left side to ensure all values fit (line 406 in tile.py)
2. For the input `[0.0, ..., 1.15625]`, the bins are calculated as `[-0.00115625, 0.57754688, 1.15625]`
3. The `precision=3` parameter then rounds these boundaries for the IntervalIndex creation
4. The maximum boundary 1.15625 gets rounded to 1.156 (3 decimal places)
5. The value 1.15625 is assigned to interval `(0.578, 1.156]` even though 1.15625 > 1.156

This contradicts the mathematical definition of interval membership. The `in` operator for intervals correctly returns `False` for `1.15625 in Interval(0.578, 1.156]`, yet the value is assigned to this interval.

## Relevant Context

The issue is in the `_format_labels()` function at line 546-577 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/reshape/tile.py`. The precision rounding is applied when creating the IntervalIndex that represents the bins:

```python
# Line 565-566
formatter = lambda x: _round_frac(x, precision)
breaks = [formatter(b) for b in bins]
```

The pandas documentation for `precision` states: "The precision at which to store and display the bins labels" - implying it should only affect display, not the actual bin assignment logic.

## Proposed Fix

The precision parameter should only affect display formatting, not the actual boundaries used for interval membership. Here's a patch that separates display formatting from actual boundaries:

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -481,10 +481,21 @@ def _bins_to_cuts(

         if labels is None:
             labels = _format_labels(
                 bins, precision, right=right, include_lowest=include_lowest
             )
+            # Create intervals with full precision for correct membership testing
+            # but display with requested precision
+            if not _is_dt_or_td(bins.dtype):
+                # Store actual bins without rounding for interval membership
+                actual_breaks = bins
+                if include_lowest and right:
+                    actual_breaks = bins.copy()
+                    actual_breaks[0] = actual_breaks[0] - 10 ** (-precision)
+                closed = "right" if right else "left"
+                # Use actual boundaries for interval creation
+                labels = IntervalIndex.from_breaks(actual_breaks, closed=closed)
+                # Apply precision formatting only for display
         elif ordered and len(set(labels)) != len(labels):
             raise ValueError(
                 "labels must be unique if ordered=True; pass ordered=False "
                 "for duplicate labels"
             )
```

Alternative simpler fix: When rounding for precision, ensure the min/max boundaries still encompass all values:

```diff
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -566,6 +566,11 @@ def _format_labels(
         adjust = lambda x: x - 10 ** (-precision)

     breaks = [formatter(b) for b in bins]
+
+    # Ensure rounded boundaries still contain all values
+    if len(breaks) > 0 and not _is_dt_or_td(bins.dtype):
+        breaks[0] = min(breaks[0], bins[0])  # floor for min
+        breaks[-1] = max(breaks[-1], bins[-1])  # ceil for max
     if right and include_lowest:
         # adjust lhs of first interval by precision to account for being right closed
         breaks[0] = adjust(breaks[0])
```