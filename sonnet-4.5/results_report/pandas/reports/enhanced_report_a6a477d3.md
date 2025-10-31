# Bug Report: pandas.core.array_algos.masked_reductions Inconsistent NA Handling for sum and prod

**Target**: `pandas.core.array_algos.masked_reductions.sum` and `pandas.core.array_algos.masked_reductions.prod`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `sum` and `prod` functions in `pandas.core.array_algos.masked_reductions` return numeric identity values (0.0 for sum, 1.0 for prod) when all values are masked, while all other reduction functions in the same module (`mean`, `min`, `max`, `var`, `std`) correctly return `NA`.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst
from pandas.core.array_algos import masked_reductions
from pandas._libs import missing as libmissing

@given(
    values=npst.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=100),
        elements=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    ),
)
@settings(max_examples=500)
def test_masked_reduction_all_masked(values):
    mask = np.ones(len(values), dtype=bool)

    assert masked_reductions.sum(values, mask, skipna=True) is libmissing.NA
    assert masked_reductions.prod(values, mask, skipna=True) is libmissing.NA
    assert masked_reductions.min(values, mask, skipna=True) is libmissing.NA
    assert masked_reductions.max(values, mask, skipna=True) is libmissing.NA
    assert masked_reductions.mean(values, mask, skipna=True) is libmissing.NA

if __name__ == "__main__":
    test_masked_reduction_all_masked()
```

<details>

<summary>
**Failing input**: `values=array([0.])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 25, in <module>
    test_masked_reduction_all_masked()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 8, in test_masked_reduction_all_masked
    values=npst.arrays(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 18, in test_masked_reduction_all_masked
    assert masked_reductions.sum(values, mask, skipna=True) is libmissing.NA
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_masked_reduction_all_masked(
    values=array([0.]),  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.array_algos import masked_reductions
from pandas._libs import missing as libmissing

values = np.array([0.])
mask = np.array([True])

print("Testing masked reductions with all values masked (mask=[True]):")
print("=" * 60)

sum_result = masked_reductions.sum(values, mask, skipna=True)
prod_result = masked_reductions.prod(values, mask, skipna=True)
mean_result = masked_reductions.mean(values, mask, skipna=True)
min_result = masked_reductions.min(values, mask, skipna=True)
max_result = masked_reductions.max(values, mask, skipna=True)
var_result = masked_reductions.var(values, mask, skipna=True)
std_result = masked_reductions.std(values, mask, skipna=True)

print(f"sum:  {sum_result} (type: {type(sum_result).__name__}) - Is NA? {sum_result is libmissing.NA}")
print(f"prod: {prod_result} (type: {type(prod_result).__name__}) - Is NA? {prod_result is libmissing.NA}")
print(f"mean: {mean_result} (type: {type(mean_result).__name__}) - Is NA? {mean_result is libmissing.NA}")
print(f"min:  {min_result} (type: {type(min_result).__name__}) - Is NA? {min_result is libmissing.NA}")
print(f"max:  {max_result} (type: {type(max_result).__name__}) - Is NA? {max_result is libmissing.NA}")
print(f"var:  {var_result} (type: {type(var_result).__name__}) - Is NA? {var_result is libmissing.NA}")
print(f"std:  {std_result} (type: {type(std_result).__name__}) - Is NA? {std_result is libmissing.NA}")

print("\nBUG SUMMARY:")
print("-" * 60)
print("sum and prod return numeric values (0.0 and 1.0) when all values are masked,")
print("while mean, min, max, var, and std correctly return NA.")
print("This is an inconsistency in the masked_reductions module.")
```

<details>

<summary>
Output shows sum returns 0.0 and prod returns 1.0 instead of NA
</summary>
```
Testing masked reductions with all values masked (mask=[True]):
============================================================
sum:  0.0 (type: float64) - Is NA? False
prod: 1.0 (type: float64) - Is NA? False
mean: <NA> (type: NAType) - Is NA? True
min:  <NA> (type: NAType) - Is NA? True
max:  <NA> (type: NAType) - Is NA? True
var:  <NA> (type: NAType) - Is NA? True
std:  <NA> (type: NAType) - Is NA? True

BUG SUMMARY:
------------------------------------------------------------
sum and prod return numeric values (0.0 and 1.0) when all values are masked,
while mean, min, max, var, and std correctly return NA.
This is an inconsistency in the masked_reductions module.
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Internal Inconsistency**: Within the same module (`masked_reductions`), different reduction functions handle the all-masked case differently. Functions `mean`, `var`, and `std` explicitly check for `mask.all()` and return NA (lines 163, 176, 194 in the source), while `sum` and `prod` do not have this check.

2. **Semantic Incorrectness**: When all values are masked (indicating missing/NA data), there is no data to aggregate. Returning identity values (0 for sum, 1 for prod) is misleading because it implies a computed result from actual data.

3. **API Inconsistency**: The module is specifically designed for "masked" reductions where masks indicate missing values. Having some functions return NA and others return numeric values for the same condition (all data masked) violates the principle of least surprise.

4. **Contradicts min_count Behavior**: When `min_count=1` is specified, sum and prod DO return NA for all-masked data, showing the capability exists but isn't applied consistently.

## Relevant Context

The masked_reductions module is located at `pandas/core/array_algos/masked_reductions.py`. The issue stems from the `_reductions` helper function (lines 26-69) which is used by `sum` and `prod` but doesn't check for the all-masked case.

In contrast:
- `mean`, `var`, `std` functions explicitly check `mask.all()` before calling `_reductions`
- `min` and `max` use a different helper `_minmax` which properly handles all-masked data by checking if the unmasked subset is empty

The current behavior appears to rely on NumPy's default behavior where `np.sum([], initial=None)` returns 0 and `np.prod([], initial=None)` returns 1, but this is inappropriate for a module explicitly handling masked/missing values.

Documentation: While pandas high-level API documentation may specify that sum returns 0 for all-NA data, this is an internal implementation module specifically for masked operations, where consistency should take precedence.

## Proposed Fix

```diff
--- a/pandas/core/array_algos/masked_reductions.py
+++ b/pandas/core/array_algos/masked_reductions.py
@@ -59,6 +59,10 @@ def _reductions(
     else:
         if check_below_min_count(values.shape, mask, min_count) and (
             axis is None or values.ndim == 1
         ):
             return libmissing.NA
+
+        # Check if all values are masked, consistent with mean/var/std
+        if mask.all():
+            return libmissing.NA

         if values.dtype == np.dtype(object):
             # object dtype does not support `where` without passing an initial
             values = values[~mask]
```