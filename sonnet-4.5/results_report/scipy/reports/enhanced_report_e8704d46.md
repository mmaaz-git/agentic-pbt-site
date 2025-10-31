# Bug Report: scipy.stats.percentileofscore Returns Values Exceeding 100

**Target**: `scipy.stats.percentileofscore`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `percentileofscore` function returns values slightly greater than 100 due to floating point arithmetic errors when the score is outside the range of all array values, violating the documented contract that percentiles should be in the range [0, 100].

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as npst
import scipy.stats as stats

@given(
    npst.arrays(
        dtype=float,
        shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=5, max_side=100),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
    ),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=300)
def test_percentileofscore_bounds(arr, score):
    percentile = stats.percentileofscore(arr, score)
    assert 0 <= percentile <= 100, \
        f"Percentile should be in [0, 100], got {percentile}"

if __name__ == "__main__":
    test_percentileofscore_bounds()
```

<details>

<summary>
**Failing input**: `arr=array([0., 0., ..., 0.]) (92 zeros)`, `score=1.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 20, in <module>
    test_percentileofscore_bounds()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 6, in test_percentileofscore_bounds
    npst.arrays(
               ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 16, in test_percentileofscore_bounds
    assert 0 <= percentile <= 100, \
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Percentile should be in [0, 100], got 100.00000000000001
Falsifying example: test_percentileofscore_bounds(
    arr=array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0.]),
    score=1.0,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/7/hypo.py:17
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.stats as stats

# Test case that causes percentileofscore to return value > 100
arr = np.array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])
score = 0.0

# Test with default 'rank' kind
print("Testing scipy.stats.percentileofscore with problematic input")
print(f"Array: {arr}")
print(f"Score: {score}")
print()

percentile = stats.percentileofscore(arr, score)
print(f"Result with kind='rank' (default): {percentile}")
print(f"Is percentile > 100? {percentile > 100}")
print(f"Exact value: {percentile:.20f}")
print()

# Test with all 'kind' parameters
for kind_param in ['rank', 'weak', 'strict', 'mean']:
    result = stats.percentileofscore(arr, score, kind=kind_param)
    print(f"Result with kind='{kind_param}': {result}")
    print(f"Is result > 100? {result > 100}")
    print(f"Exact value: {result:.20f}")
    print()
```

<details>

<summary>
percentileofscore returns 100.00000000000001 for all 'kind' parameters
</summary>
```
Testing scipy.stats.percentileofscore with problematic input
Array: [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.]
Score: 0.0

Result with kind='rank' (default): 100.00000000000001
Is percentile > 100? True
Exact value: 100.00000000000001421085

Result with kind='rank': 100.00000000000001
Is result > 100? True
Exact value: 100.00000000000001421085

Result with kind='weak': 100.00000000000001
Is result > 100? True
Exact value: 100.00000000000001421085

Result with kind='strict': 100.00000000000001
Is result > 100? True
Exact value: 100.00000000000001421085

Result with kind='mean': 100.00000000000001
Is result > 100? True
Exact value: 100.00000000000001421085
```
</details>

## Why This Is A Bug

According to the scipy.stats.percentileofscore documentation (line 2152 of _stats_py.py), the function should return "Percentile-position of score (0-100) relative to `a`." By mathematical definition, percentiles represent percentages and must be in the range [0, 100]. A percentile cannot exceed 100% as that would mean more than 100% of values fall below the given score, which is mathematically impossible.

The bug occurs due to floating point arithmetic precision errors in the calculation. When the score is greater than all values in the array, the function calculates `count(a <= score) * (100.0 / n)`. For example, with 11 elements all less than the score, this becomes `11 * (100.0 / 11)`, which due to floating point division produces 100.00000000000001421085 instead of exactly 100.0.

This violates the API contract and could cause failures in downstream code that validates percentile ranges (e.g., `assert percentile <= 100`), visualization libraries expecting strict [0, 100] bounds, or statistical computations that depend on this invariant.

## Relevant Context

The bug affects all four 'kind' parameters ('rank', 'weak', 'strict', 'mean') because they all use similar arithmetic patterns involving division by `n` and multiplication by constants like 50.0 or 100.0.

The issue is located in scipy/stats/_stats_py.py lines 2240-2255, where the percentile calculations are performed:
- For 'rank': `perct = (left + right + plus1) * (50.0 / n)`
- For 'weak': `perct = count(a <= score) * (100.0 / n)`
- For 'strict': `perct = count(a < score) * (100.0 / n)`
- For 'mean': `perct = (left + right) * (50.0 / n)`

Similar functions like numpy.percentile and scipy.stats.scoreatpercentile properly enforce the [0, 100] range constraint. The documentation even shows an example (line 2196-2197) where `np.inf` returns exactly `100.` rather than exceeding it.

Documentation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.percentileofscore.html

## Proposed Fix

```diff
--- a/scipy/stats/_stats_py.py
+++ b/scipy/stats/_stats_py.py
@@ -2257,8 +2257,9 @@ def percentileofscore(a, score, kind='rank', nan_policy='propagate'):
     # Re-insert nan values
     perct = ma.filled(perct, np.nan)

+    # Ensure result stays within [0, 100] range to handle floating point errors
+    perct = np.clip(perct, 0.0, 100.0)
     if perct.ndim == 0:
         return perct[()]
     return perct
```