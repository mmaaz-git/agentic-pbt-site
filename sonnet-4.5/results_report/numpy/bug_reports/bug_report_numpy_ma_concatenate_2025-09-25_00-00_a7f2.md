# Bug Report: numpy.ma.concatenate Mask Type Inconsistency

**Target**: `numpy.ma.concatenate`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When concatenating empty masked arrays, `numpy.ma.concatenate` returns a result with a scalar `numpy.bool` mask instead of an empty ndarray mask, breaking the invariant that mask type should match data shape.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import numpy.ma as ma

@given(
    st.lists(st.integers(), min_size=0, max_size=20),
    st.lists(st.booleans(), min_size=0, max_size=20),
    st.lists(st.integers(), min_size=0, max_size=20),
    st.lists(st.booleans(), min_size=0, max_size=20),
)
def test_concatenate_preserves_length(data1, mask1, data2, mask2):
    assume(len(data1) == len(mask1))
    assume(len(data2) == len(mask2))

    arr1 = ma.masked_array(data1, mask=mask1)
    arr2 = ma.masked_array(data2, mask=mask2)

    result = ma.concatenate([arr1, arr2])

    assert len(result) == len(arr1) + len(arr2)
    assert len(result.mask) == len(arr1) + len(arr2)
```

**Failing input**: `data1=[], mask1=[], data2=[], mask2=[]`

## Reproducing the Bug

```python
import numpy.ma as ma

arr1 = ma.masked_array([], mask=[])
arr2 = ma.masked_array([], mask=[])

print(f"arr1.mask: {arr1.mask}, type: {type(arr1.mask).__name__}")
print(f"arr2.mask: {arr2.mask}, type: {type(arr2.mask).__name__}")

result = ma.concatenate([arr1, arr2])

print(f"result.mask: {result.mask}, type: {type(result.mask).__name__}")
print(f"result.data: {result.data}, type: {type(result.data).__name__}")

try:
    len(result.mask)
except TypeError as e:
    print(f"ERROR calling len(result.mask): {e}")

try:
    result.mask[0]
except (TypeError, IndexError) as e:
    print(f"ERROR indexing result.mask[0]: {e}")
```

## Why This Is A Bug

The `mask` attribute should consistently be an ndarray matching the shape of the data array. When concatenating empty masked arrays:
- Input masks are both empty ndarrays: `[]`
- Result data is an empty ndarray: `array([], dtype=float64)`
- But result mask is a scalar: `False` (numpy.bool)

This type inconsistency violates the masked array contract and breaks operations like `len(result.mask)` and `result.mask[i]`. Code that works with non-empty concatenation results will fail with empty ones.

## Fix

The fix should ensure that when concatenating arrays results in an empty array, the mask remains an empty ndarray rather than being converted to a scalar `False`. This likely requires checking for empty results in the `concatenate` implementation and preserving the array type for the mask.

```diff
The exact location would need to be identified in numpy/ma/core.py around the concatenate function implementation. The fix would involve detecting when the result is empty and ensuring:
  result.mask = np.array([], dtype=bool)
instead of:
  result.mask = False
```