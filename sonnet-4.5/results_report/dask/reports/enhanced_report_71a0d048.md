# Bug Report: dask.array.eye Incorrect Chunking for Non-Square Matrices

**Target**: `dask.array.eye`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`dask.array.eye` crashes with a "Missing dependency" error when creating non-square identity matrices (N != M) with chunk size >= M due to incorrect chunk specification in the Array constructor.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import dask.array as da
import numpy as np

@given(
    st.integers(min_value=2, max_value=10),
    st.integers(min_value=2, max_value=10),
    st.integers(min_value=-1, max_value=1)
)
@settings(max_examples=300, deadline=None)
def test_eye_diagonal_ones(N, M, k):
    """
    Property: eye creates identity matrix with ones on diagonal
    Evidence: eye creates matrix with 1s on main diagonal
    """
    assume(N > abs(k) and M > abs(k))

    arr = da.eye(N, chunks=3, M=M, k=k)
    computed = arr.compute()

    for i in range(N):
        for j in range(M):
            if j - i == k:
                assert computed[i, j] == 1.0
            else:
                assert computed[i, j] == 0.0

if __name__ == "__main__":
    test_eye_diagonal_ones()
```

<details>

<summary>
**Failing input**: `N=2, M=3, k=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 29, in <module>
    test_eye_diagonal_ones()
    ~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 6, in test_eye_diagonal_ones
    st.integers(min_value=2, max_value=10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 19, in test_eye_diagonal_ones
    computed = arr.compute()
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 373, in compute
    (result,) = compute(self, traverse=False, **kwargs)
                ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/base.py", line 681, in compute
    results = schedule(expr, keys, **kwargs)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/local.py", line 191, in start_state_from_dask
    raise ValueError(
    ...<3 lines>...
    )
ValueError: Missing dependency ('eye-b9a630cc5f44363256f427428c37836c', 0, 1) for dependents {'finalize-hlgfinalizecompute-6f206b6d7ac0478ead46ebf1b8fae5a9'}
Falsifying example: test_eye_diagonal_ones(
    N=2,
    M=3,
    k=0,
)
```
</details>

## Reproducing the Bug

```python
import dask.array as da
import numpy as np

print("Testing dask.array.eye with non-square matrix and chunks >= M")
print("=" * 60)

# Show that NumPy works correctly
print("\n1. NumPy baseline (works correctly):")
np_result = np.eye(2, M=3)
print(f"   np.eye(2, M=3) shape: {np_result.shape}")
print(f"   Result:\n{np_result}")

# Show the failing case
print("\n2. Dask failing case: da.eye(2, chunks=3, M=3, k=0)")
try:
    arr = da.eye(2, chunks=3, M=3, k=0)
    print(f"   Created array with shape {arr.shape}")
    print("   Attempting to compute...")
    result = arr.compute()
    print(f"   Success! Result:\n{result}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")

# Show working cases for comparison
print("\n3. Dask working case 1: da.eye(2, chunks=2, M=3) [chunks < M]")
try:
    arr = da.eye(2, chunks=2, M=3)
    print(f"   Created array with shape {arr.shape}")
    result = arr.compute()
    print(f"   Success! Result:\n{result}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")

print("\n4. Dask working case 2: da.eye(3, chunks=3, M=3) [square matrix]")
try:
    arr = da.eye(3, chunks=3, M=3)
    print(f"   Created array with shape {arr.shape}")
    result = arr.compute()
    print(f"   Success! Result:\n{result}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError: Missing dependency for non-square matrix with chunks >= M
</summary>
```
Testing dask.array.eye with non-square matrix and chunks >= M
============================================================

1. NumPy baseline (works correctly):
   np.eye(2, M=3) shape: (2, 3)
   Result:
[[1. 0. 0.]
 [0. 1. 0.]]

2. Dask failing case: da.eye(2, chunks=3, M=3, k=0)
   Created array with shape (2, 3)
   Attempting to compute...
   ERROR: ValueError: Missing dependency ('eye-b9a630cc5f44363256f427428c37836c', 0, 1) for dependents {'finalize-hlgfinalizecompute-30b1b4676dcc4e899b6a76a9bde65e3f'}

3. Dask working case 1: da.eye(2, chunks=2, M=3) [chunks < M]
   Created array with shape (2, 3)
   Success! Result:
[[1. 0. 0.]
 [0. 1. 0.]]

4. Dask working case 2: da.eye(3, chunks=3, M=3) [square matrix]
   Created array with shape (3, 3)
   Success! Result:
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Documentation explicitly supports non-square matrices**: The `eye` function documentation clearly states that the `M` parameter allows creation of non-square identity matrices, with the return type specified as "Array of shape (N,M)".

2. **NumPy compatibility broken**: Dask arrays claim NumPy compatibility. NumPy's `eye` function correctly handles `np.eye(2, M=3)`, producing a 2x3 matrix with ones on the diagonal. The same parameters cause Dask to crash.

3. **Chunking should not affect correctness**: The chunks parameter is meant for performance optimization and memory management, not to restrict functionality. There's no documented restriction that chunks must be less than M for non-square matrices.

4. **Implementation error is clear**: The code correctly computes `vchunks` and `hchunks` via `normalize_chunks()` on line 602, accounting for the (N, M) shape. However, line 624 incorrectly uses `chunks=(chunks, chunks)` where `chunks=vchunks[0]`, instead of using `chunks=(vchunks, hchunks)`. This causes a mismatch between the task graph creation (which correctly uses both vchunks and hchunks) and the Array metadata (which incorrectly uses the same value for both dimensions).

## Relevant Context

The bug manifests specifically when:
- Creating non-square matrices (N != M)
- The chunk size parameter results in different chunking patterns for rows vs columns
- Most commonly when chunks >= M

The error message "Missing dependency" is misleading - the actual issue is that the Array constructor is told the wrong chunk structure. Tasks are created for the correct chunk pattern (based on vchunks and hchunks), but the Array is told to expect a different pattern (based on chunks, chunks).

Code location: `/dask/array/creation.py`, function `eye`, line 624

Documentation reference: The eye function docstring explicitly states support for non-square matrices via the M parameter.

## Proposed Fix

```diff
--- a/dask/array/creation.py
+++ b/dask/array/creation.py
@@ -621,7 +621,7 @@ def eye(N, chunks="auto", M=None, k=0, dtype=float):
             else:
                 t = Task(key, np.zeros, (vchunk, hchunk), dtype)
             dsk[t.key] = t
-    return Array(dsk, name_eye, shape=(N, M), chunks=(chunks, chunks), dtype=dtype)
+    return Array(dsk, name_eye, shape=(N, M), chunks=(vchunks, hchunks), dtype=dtype)


 @derived_from(np)
```