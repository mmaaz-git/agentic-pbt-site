# Bug Report: scipy.ndimage.sum Excludes Label 0 When index=None

**Target**: `scipy.ndimage.sum` (alias for `scipy.ndimage.sum_labels`)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `scipy.ndimage.sum` is called with `index=None` on data containing label 0 (background), it incorrectly excludes pixels with label 0 from the sum, violating the documented behavior and user expectations.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as npst
import numpy as np
import scipy.ndimage as ndi

@given(
    labels=npst.arrays(
        dtype=np.int32,
        shape=npst.array_shapes(min_dims=1, max_dims=2, min_side=3, max_side=10),
        elements=st.integers(min_value=0, max_value=5)
    ),
    values=npst.arrays(
        dtype=np.float64,
        shape=npst.array_shapes(min_dims=1, max_dims=2, min_side=3, max_side=10),
        elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
    )
)
def test_labeled_sum_consistency(labels, values):
    assume(labels.shape == values.shape)

    total_sum = ndi.sum(values, labels=labels, index=None)
    unique_labels = np.unique(labels)
    individual_sums = ndi.sum(values, labels=labels, index=list(unique_labels))

    if np.isscalar(individual_sums):
        individual_sums = np.array([individual_sums])

    expected_total = np.sum(individual_sums)

    np.testing.assert_allclose(total_sum, expected_total, rtol=1e-10, atol=1e-10)
```

**Failing input**: `labels=array([0, 1, 1], dtype=int32)`, `values=array([1., 1., 1.])`

## Reproducing the Bug

```python
import numpy as np
import scipy.ndimage as ndi

labels = np.array([0, 1, 1], dtype=np.int32)
values = np.array([1., 1., 1.])

total_with_none = ndi.sum(values, labels=labels, index=None)
print(f"ndi.sum(values, labels, index=None) = {total_with_none}")

sum_label_0 = ndi.sum(values, labels=labels, index=0)
sum_label_1 = ndi.sum(values, labels=labels, index=1)
print(f"ndi.sum(values, labels, index=0) = {sum_label_0}")
print(f"ndi.sum(values, labels, index=1) = {sum_label_1}")

print(f"Sum of all labels: {sum_label_0 + sum_label_1}")
print(f"np.sum(values): {np.sum(values)}")
```

**Output:**
```
ndi.sum(values, labels, index=None) = 2.0
ndi.sum(values, labels, index=0) = 1.0
ndi.sum(values, labels, index=1) = 2.0
Sum of all labels: 3.0
np.sum(values): 3.0
```

**Expected:** `index=None` should return 3.0 (sum of all values)
**Actual:** Returns 2.0 (excludes label 0)

## Why This Is A Bug

1. **Inconsistency with documentation**: The docstring states "Calculate the sum of the values of the array" without mentioning that label 0 is excluded.

2. **Unintuitive behavior**: When `index=None`, users expect either:
   - Sum of all values in the input array (like `np.sum(input)`), OR
   - Sum across all unique labels (equivalent to `index=np.unique(labels)`)

3. **Internal inconsistency**:
   - `ndi.sum(values, labels, index=[0,1])` returns the sum for labels 0 and 1
   - `ndi.sum(values, labels, index=None)` excludes label 0

   This violates the principle that `None` should mean "all" in NumPy/SciPy APIs.

4. **No documented rationale**: There is no clear reason why label 0 should be special when `index=None`, especially since explicitly including 0 in the index works fine.

## Fix

The bug appears to be in the `_stats` function called by `sum_labels`. When `index=None`, the function should include label 0 in the computation.

**Expected behavior options:**

**Option 1** (Recommended): When `index=None`, sum ALL values
```python
if index is None:
    return np.sum(input)
```

**Option 2**: When `index=None` with labels, sum all unique labels
```python
if index is None and labels is not None:
    index = np.unique(labels)
```

**Option 3**: Document the current behavior clearly
If the current behavior is intentional (excluding label 0), this MUST be documented prominently in the docstring with a clear rationale.

Given that label 0 is conventionally used as "background" in image processing, Option 2 makes the most sense for labeled operations, ensuring consistency with explicit index specification.