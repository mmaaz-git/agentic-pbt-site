# Bug Report: scipy.stats.binned_statistic Excludes Values at Rightmost Bin Edge

**Target**: `scipy.stats.binned_statistic`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.stats.binned_statistic` incorrectly excludes values that exactly equal the rightmost bin edge, assigning them a bin number outside the valid range. This causes data loss when counting or aggregating statistics.

## Property-Based Test

```python
import numpy as np
import scipy.stats as stats
from hypothesis import given, strategies as st, assume, settings


@given(
    data=st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=50
    ),
    bins=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=1000)
def test_binned_statistic_counts_all_data(data, bins):
    """Property: binned_statistic should count all finite input values"""
    arr = np.array(data)
    assume(len(np.unique(arr)) > 1)

    result = stats.binned_statistic(arr, arr, statistic='count', bins=bins)

    total_count = np.sum(result.statistic)
    expected_count = len(arr)

    assert total_count == expected_count, \
        f"binned_statistic lost {expected_count - total_count} values: " \
        f"data={arr}, bins={bins}, binnumber={result.binnumber}"
```

**Failing input**: `data=[0.0, 0.0, 0.0, 0.0, 1.1125369292536007e-308], bins=2`

## Reproducing the Bug

```python
import numpy as np
import scipy.stats as stats

data = [0.0, 0.0, 0.0, 0.0, 1.1125369292536007e-308]
arr = np.array(data)

result = stats.binned_statistic(arr, arr, statistic='count', bins=2)

print(f"Input data: {data}")
print(f"Number of data points: {len(data)}")
print(f"Bin edges: {result.bin_edges}")
print(f"Bin numbers: {result.binnumber}")
print(f"Counts per bin: {result.statistic}")
print(f"Total counted: {np.sum(result.statistic)}")

assert np.sum(result.statistic) == len(data), \
    f"Lost {len(data) - np.sum(result.statistic)} values!"
```

**Output:**
```
Input data: [0.0, 0.0, 0.0, 0.0, 1.1125369292536007e-308]
Number of data points: 5
Bin edges: [0.00000000e+000 5.56268465e-309 1.11253693e-308]
Bin numbers: [1 1 1 1 3]
Counts per bin: [4. 0.]
Total counted: 4.0
AssertionError: Lost 1.0 values!
```

The last value (1.11e-308) equals the rightmost bin edge and is assigned bin number 3, which is outside the valid range [1, 2].

## Why This Is A Bug

1. **Violates fundamental property**: All finite input values should be assigned to a bin. The function silently loses data.

2. **Inconsistent with numpy.histogram**: `np.histogram` treats the rightmost bin as closed `[a, b]`, ensuring all values are counted. Users expect the same behavior from `binned_statistic`.

3. **Undocumented behavior**: The documentation doesn't mention that values at the rightmost edge will be excluded.

4. **Impact**: This causes incorrect results in statistical analyses where the maximum value equals a bin edge. The bug is silent - no error is raised, making it hard to detect.

## Fix

The root cause is in the bin assignment logic. The function uses `np.digitize` (or equivalent) with `right=False`, treating all bins as half-open `[a, b)`. However, the rightmost bin should be closed `[a, b]` to include the maximum value.

The fix should special-case values equal to the rightmost edge:

```diff
--- a/scipy/stats/_binned_statistic.py
+++ b/scipy/stats/_binned_statistic.py
@@ -somewhere in binned_statistic_dd
     # Current code (simplified):
     Vdigitized = np.digitize(V, bins, right=False)
+
+    # Fix: Include values at rightmost edge in the last bin
+    Vdigitized[V == bins[-1]] = len(bins) - 1
```

This ensures that any value exactly equal to the rightmost bin edge is placed in the last bin instead of being excluded.