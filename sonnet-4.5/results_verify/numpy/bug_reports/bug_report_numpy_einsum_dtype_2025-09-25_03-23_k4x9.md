# Bug Report: numpy.einsum dtype inconsistency with sum/trace

**Target**: `numpy.einsum`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.einsum` performs reduction operations without dtype promotion, causing silent integer overflow, while equivalent numpy functions (`sum`, `trace`) correctly promote int32 to int64 to preserve accuracy.

## Property-Based Test

```python
import numpy as np
from hypothesis import given
from hypothesis.extra import numpy as npst

@given(npst.arrays(dtype=np.int32, shape=(5,)))
def test_einsum_sum_equivalence(arr):
    einsum_sum = np.einsum('i->', arr)
    np_sum = np.sum(arr)
    assert einsum_sum == np_sum
```

**Failing input**: `array([429496730, 429496730, 429496730, 429496730, 429496730], dtype=int32)`

## Reproducing the Bug

```python
import numpy as np

arr = np.array([429496730] * 5, dtype=np.int32)

einsum_result = np.einsum('i->', arr)
sum_result = np.sum(arr)

print(f"einsum: {einsum_result} (dtype: {einsum_result.dtype})")
print(f"sum:    {sum_result} (dtype: {sum_result.dtype})")
```

Output:
```
einsum: -2147483646 (dtype: int32)
sum:    2147483650 (dtype: int64)
```

The same issue affects trace:
```python
matrix = np.array([[715827883] * 3] * 3, dtype=np.int32)

einsum_trace = np.einsum('ii->', matrix)
np_trace = np.trace(matrix)

print(f"einsum: {einsum_trace} (dtype: {einsum_trace.dtype})")
print(f"trace:  {np_trace} (dtype: {np_trace.dtype})")
```

Output:
```
einsum: -2147483647 (dtype: int32)
trace:  2147483649 (dtype: int64)
```

## Why This Is A Bug

`numpy.einsum` and standard reduction functions (`sum`, `trace`, etc.) should have consistent overflow handling. When performing reductions on int32 arrays that could overflow, numpy's standard functions automatically promote to int64, but `einsum` silently overflows while preserving int32, leading to incorrect results.

This inconsistency violates the principle of least surprise: users performing mathematically equivalent operations (`einsum('i->', arr)` vs `sum(arr)`) expect the same result, not silent data corruption in one case.

## Fix

The einsum function should use the same dtype promotion rules as other numpy reduction operations. When the output could overflow the input dtype during accumulation, it should automatically promote to a larger dtype (e.g., int32 → int64).

The fix would likely be in the einsum implementation to check if the operation is a reduction and apply appropriate dtype promotion, similar to how `sum`, `trace`, `prod`, and other reductions handle it.

Workaround: Users can explicitly specify `dtype=np.int64` when using einsum for reductions: `np.einsum('i->', arr, dtype=np.int64)`.