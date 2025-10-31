# Bug Report: dask.array.slicing normalize_slice Empty Slice Mishandling

**Target**: `dask.array.slicing.normalize_slice`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `normalize_slice` function incorrectly transforms empty slices with negative indices and negative steps into non-empty slices, causing Dask arrays to return different results than NumPy arrays for certain edge case slicing operations.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import assume, given, settings, strategies as st
from dask.array.slicing import normalize_slice


@given(
    st.integers(min_value=-20, max_value=20),
    st.integers(min_value=-20, max_value=20),
    st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0),
    st.integers(min_value=1, max_value=100)
)
@settings(max_examples=500)
def test_normalize_slice_equivalence(start, stop, step, dim):
    assume(dim > 0)
    arr = np.arange(dim)
    idx = slice(start, stop, step)

    normalized = normalize_slice(idx, dim)

    original_result = arr[idx]
    normalized_result = arr[normalized]

    assert np.array_equal(original_result, normalized_result), \
        f"Normalized slice should produce same elements: {idx} vs {normalized} on array of length {dim}"

if __name__ == "__main__":
    test_normalize_slice_equivalence()
```

<details>

<summary>
**Failing input**: `start=-2, stop=-2, step=-1, dim=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 30, in <module>
    test_normalize_slice_equivalence()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 10, in test_normalize_slice_equivalence
    st.integers(min_value=-20, max_value=20),
               ^^^
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 26, in test_normalize_slice_equivalence
    assert np.array_equal(original_result, normalized_result), \
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Normalized slice should produce same elements: slice(-2, -2, -1) vs slice(-1, None, -1) on array of length 1
Falsifying example: test_normalize_slice_equivalence(
    start=-2,
    stop=-2,
    step=-1,
    dim=1,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/numeric.py:2588
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import numpy as np
import dask.array as da

arr = np.array([0])
darr = da.from_array(arr, chunks=-1)

idx = slice(-2, -2, -1)

np_result = arr[idx]
dask_result = darr[idx].compute()

print(f"NumPy: arr[{idx}] = {np_result}")
print(f"Dask:  darr[{idx}].compute() = {dask_result}")

assert np.array_equal(np_result, dask_result), f"Results differ: NumPy returned {np_result}, Dask returned {dask_result}"
```

<details>

<summary>
AssertionError: Results differ - NumPy returns empty array, Dask returns [0]
</summary>
```
NumPy: arr[slice(-2, -2, -1)] = []
Dask:  darr[slice(-2, -2, -1)].compute() = [0]
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/46/repo.py", line 18, in <module>
    assert np.array_equal(np_result, dask_result), f"Results differ: NumPy returned {np_result}, Dask returned {dask_result}"
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Results differ: NumPy returned [], Dask returned [0]
```
</details>

## Why This Is A Bug

This violates expected behavior because Dask arrays are designed to be drop-in replacements for NumPy arrays, yet they produce different results for the same slicing operations.

When `slice(-2, -2, -1)` is applied to a 1-dimensional array of length 1:
1. The slice's `indices(1)` method correctly normalizes this to `(-1, -1, -1)`
2. NumPy correctly interprets `slice(-1, -1, -1)` as an empty slice (start equals stop)
3. However, `normalize_slice` then applies additional transformations in lines 791-795 that break this semantic equivalence
4. Specifically, when `step < 0` and `stop < 0`, the function sets `stop = None` (line 795)
5. This transforms `slice(-1, -1, -1)` into `slice(-1, None, -1)`
6. The transformed slice now means "from index -1 to the beginning" instead of "empty slice"
7. This causes `arr[slice(-1, None, -1)]` to return `[0]` instead of an empty array

Python's slicing semantics are well-established: any slice where start equals stop should produce an empty result, regardless of the step value. This is consistent across Python lists, NumPy arrays, and other sequence types. The `normalize_slice` function violates this fundamental rule by changing the semantic meaning of the slice rather than just optimizing its representation.

## Relevant Context

The bug occurs in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/array/slicing.py` at lines 791-795. The function's docstring states it should "Normalize slices to canonical form" but does not explicitly define what canonical form means or whether semantic equivalence should be preserved.

There is an existing open GitHub issue (#10555) that reports this exact bug, confirming it's recognized as unexpected behavior that deviates from NumPy compatibility.

The `normalize_slice` function appears to be an internal utility function used throughout Dask's array slicing implementation to optimize slice representations (e.g., converting `slice(0, 10, 1)` to `slice(None, None, None)` for a length-10 array). However, the current implementation incorrectly assumes that setting `stop=None` for negative stops with negative steps is always safe, without considering the edge case where the slice was originally empty.

## Proposed Fix

```diff
--- a/dask/array/slicing.py
+++ b/dask/array/slicing.py
@@ -789,6 +789,9 @@ def normalize_slice(idx, dim):
             if stop is not None and start is not None and stop < start:
                 stop = start
         elif step < 0:
+            # Preserve empty slices (where start == stop)
+            if start == stop:
+                return slice(start, stop, step)
             if start >= dim - 1:
                 start = None
             if stop < 0:
```