# Bug Report: xarray.core.nputils.inverse_permutation Accepts Sentinel Values as Input

**Target**: `xarray.core.nputils.inverse_permutation`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `inverse_permutation` function uses -1 as a sentinel value to indicate positions not in the permutation, but does not validate that input indices are non-negative. When the output (containing -1) is used as input, numpy's negative indexing treats -1 as a valid index, producing incorrect results and breaking the expected mathematical property that applying the inverse operation twice should return to the original.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import given, strategies as st, settings
from xarray.core.nputils import inverse_permutation


@given(st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=50, unique=True))
@settings(max_examples=1000)
def test_inverse_permutation_involution(indices_list):
    indices = np.array(indices_list, dtype=np.intp)
    N = np.max(indices) + 1

    inv1 = inverse_permutation(indices, N)
    inv2 = inverse_permutation(inv1, N)

    assert np.array_equal(inv2, indices), f"Double inverse should return original: {indices} -> {inv1} -> {inv2}"
```

**Failing input**: `indices_list=[1]`

## Reproducing the Bug

```python
import numpy as np
from xarray.core.nputils import inverse_permutation

indices = np.array([1], dtype=np.intp)
N = 2

inv1 = inverse_permutation(indices, N)
print(f"inv1: {inv1}")

inv2 = inverse_permutation(inv1, N)
print(f"inv2: {inv2}")

print(f"inv2 == indices? {np.array_equal(inv2, indices)}")
print(f"Expected: {indices}")
print(f"Got: {inv2}")
```

Output:
```
inv1: [-1  0]
inv2: [1 0]
inv2 == indices? False
Expected: [1]
Got: [1 0]
```

## Why This Is A Bug

The `inverse_permutation` function is designed to compute the inverse of a permutation. For partial permutations, it returns -1 for positions not in the permutation (as seen in `xarray.core.groupby._inverse_permutation_indices` at line 177, which filters these out).

However, the function does not validate that its inputs are non-negative. When an array containing -1 is passed as input:
1. The line `inverse_permutation[indices] = np.arange(len(indices), dtype=np.intp)` uses the indices directly
2. Numpy's negative indexing treats -1 as the last element
3. This produces mathematically incorrect results

The function name and purpose suggest it should be involutive for complete permutations (i.e., applying it twice should return the original). This property is violated because the sentinel value -1 leaks into the computation.

## Fix

Add input validation to ensure indices are non-negative:

```diff
def inverse_permutation(indices: np.ndarray, N: int | None = None) -> np.ndarray:
    """Return indices for an inverse permutation.

    Parameters
    ----------
    indices : 1D np.ndarray with dtype=int
        Integer positions to assign elements to.
    N : int, optional
        Size of the array

    Returns
    -------
    inverse_permutation : 1D np.ndarray with dtype=int
        Integer indices to take from the original array to create the
        permutation.
    """
+   if np.any(indices < 0):
+       raise ValueError(f"indices must be non-negative, got {indices}")
    if N is None:
        N = len(indices)
    # use intp instead of int64 because of windows :(
    inverse_permutation = np.full(N, -1, dtype=np.intp)
    inverse_permutation[indices] = np.arange(len(indices), dtype=np.intp)
    return inverse_permutation
```

This fix ensures that the sentinel value -1 (which is part of the output) cannot be accidentally used as input, preventing the numpy negative indexing issue and making the function's behavior more predictable and mathematically correct.