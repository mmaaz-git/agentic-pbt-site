# Bug Report: xarray.core.nputils.inverse_permutation Silently Corrupts Data When Fed Its Own Output

**Target**: `xarray.core.nputils.inverse_permutation`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `inverse_permutation` function returns -1 as a sentinel value for positions not in the permutation, but when this output is used as input, numpy's negative indexing interprets -1 as the last array element, causing silent data corruption and breaking the mathematical involution property.

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
    """Test that applying inverse_permutation twice returns to the original for partial permutations."""
    indices = np.array(indices_list, dtype=np.intp)
    N = np.max(indices) + 1

    inv1 = inverse_permutation(indices, N)
    inv2 = inverse_permutation(inv1, N)

    assert np.array_equal(inv2, indices), f"Double inverse should return original: {indices} -> {inv1} -> {inv2}"

if __name__ == "__main__":
    test_inverse_permutation_involution()
```

<details>

<summary>
**Failing input**: `indices_list=[1]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 22, in <module>
    test_inverse_permutation_involution()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 10, in test_inverse_permutation_involution
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 19, in test_inverse_permutation_involution
    assert np.array_equal(inv2, indices), f"Double inverse should return original: {indices} -> {inv1} -> {inv2}"
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
AssertionError: Double inverse should return original: [1] -> [-1  0] -> [1 0]
Falsifying example: test_inverse_permutation_involution(
    indices_list=[1],
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from xarray.core.nputils import inverse_permutation

indices = np.array([1], dtype=np.intp)
N = 2

print("Testing inverse_permutation with partial permutation:")
print(f"Initial indices: {indices}")
print(f"N (size of output array): {N}")
print()

inv1 = inverse_permutation(indices, N)
print(f"First inverse (inv1): {inv1}")
print(f"Note: inv1[0] = -1 (sentinel value indicating position 0 not in original permutation)")
print(f"      inv1[1] = 0 (element at position 0 in original goes to position 1)")
print()

inv2 = inverse_permutation(inv1, N)
print(f"Second inverse (inv2): {inv2}")
print(f"Note: When inv1=[-1, 0] is used as input:")
print(f"      The -1 is treated as a valid index by numpy (negative indexing)")
print(f"      inv2[-1] gets set to 0 (i.e., inv2[1] = 0)")
print(f"      inv2[0] gets set to 1")
print()

print(f"Expected result: {indices} (should be involutive)")
print(f"Actual result: {inv2}")
print(f"Are they equal? {np.array_equal(inv2, indices)}")
print()

print("Why this is wrong:")
print("1. The function uses -1 as a sentinel value to mark 'not in permutation'")
print("2. But it doesn't validate that input indices are non-negative")
print("3. When -1 appears in input, numpy treats it as index from end (last element)")
print("4. This breaks the mathematical property of involution for partial permutations")
```

<details>

<summary>
AssertionError: inverse_permutation is not involutive for partial permutations
</summary>
```
Testing inverse_permutation with partial permutation:
Initial indices: [1]
N (size of output array): 2

First inverse (inv1): [-1  0]
Note: inv1[0] = -1 (sentinel value indicating position 0 not in original permutation)
      inv1[1] = 0 (element at position 0 in original goes to position 1)

Second inverse (inv2): [1 0]
Note: When inv1=[-1, 0] is used as input:
      The -1 is treated as a valid index by numpy (negative indexing)
      inv2[-1] gets set to 0 (i.e., inv2[1] = 0)
      inv2[0] gets set to 1

Expected result: [1] (should be involutive)
Actual result: [1 0]
Are they equal? False

Why this is wrong:
1. The function uses -1 as a sentinel value to mark 'not in permutation'
2. But it doesn't validate that input indices are non-negative
3. When -1 appears in input, numpy treats it as index from end (last element)
4. This breaks the mathematical property of involution for partial permutations
```
</details>

## Why This Is A Bug

The `inverse_permutation` function computes the inverse of a permutation mapping. For partial permutations (where not all positions are specified), it uses -1 as a sentinel value to mark positions that are "not in the permutation". This design choice is documented in the function's behavior and is relied upon by other parts of xarray - for example, `xarray.core.groupby._inverse_permutation_indices` (line 177) explicitly filters out these -1 values after calling the function.

However, the function fails to validate that input indices are non-negative. When an array containing -1 sentinel values (which is the function's own output format) is passed back as input, numpy's array indexing interprets -1 as a valid negative index referring to the last element of the array. This causes:

1. **Silent data corruption**: The line `inverse_permutation[indices] = np.arange(len(indices), dtype=np.intp)` assigns values to incorrect positions when indices contains -1
2. **Broken mathematical properties**: For permutations, applying the inverse operation twice should return the original permutation (involution property), but this fails
3. **Inconsistent behavior**: The function produces arrays with -1 values but cannot correctly process such arrays as input

## Relevant Context

The function is located in `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/core/nputils.py` at lines 73-94.

The function is used in several places within xarray:
- `xarray.core.indexes.py`: Uses it for index manipulation
- `xarray.core.variable.py`: Uses it for variable reordering
- `xarray.core.groupby.py`: Has a wrapper `_inverse_permutation_indices` that filters out -1 values (line 177), showing awareness of this sentinel value issue

The existing code in groupby.py demonstrates that xarray developers are aware that -1 values need special handling, but this validation is done at the call site rather than within the function itself.

## Proposed Fix

Add input validation to ensure indices are non-negative, preventing the sentinel value -1 from being misinterpreted:

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
+   # Validate that indices are non-negative to prevent sentinel value -1
+   # from being treated as a valid negative index
+   if np.any(indices < 0):
+       raise ValueError(f"indices must be non-negative, got min value: {np.min(indices)}")
    if N is None:
        N = len(indices)
    # use intp instead of int64 because of windows :(
    inverse_permutation = np.full(N, -1, dtype=np.intp)
    inverse_permutation[indices] = np.arange(len(indices), dtype=np.intp)
    return inverse_permutation
```