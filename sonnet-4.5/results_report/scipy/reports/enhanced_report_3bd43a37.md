# Bug Report: scipy.sparse.eye_array ValueError for Out-of-Bounds Diagonals

**Target**: `scipy.sparse.eye_array`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.sparse.eye_array(n, k=k)` raises a `ValueError` when the diagonal offset `k` is completely out of bounds (`abs(k) >= n` for square matrices), while `numpy.eye(n, k=k)` gracefully returns a zero matrix for the same inputs. This inconsistency violates the expected behavioral equivalence between sparse and dense array implementations.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.sparse as sp


@given(
    n=st.integers(min_value=1, max_value=20),
    k=st.integers(min_value=-5, max_value=5)
)
@settings(max_examples=200)
def test_eye_matches_dense(n, k):
    sparse_eye = sp.eye_array(n, k=k).toarray()
    dense_eye = np.eye(n, k=k)

    np.testing.assert_array_equal(
        sparse_eye,
        dense_eye,
        err_msg="eye_array doesn't match numpy.eye"
    )


if __name__ == "__main__":
    # Run the test
    test_eye_matches_dense()
```

<details>

<summary>
**Failing input**: `n=1, k=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 24, in <module>
    test_eye_matches_dense()
    ~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 7, in test_eye_matches_dense
    n=st.integers(min_value=1, max_value=20),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 12, in test_eye_matches_dense
    sparse_eye = sp.eye_array(n, k=k).toarray()
                 ~~~~~~~~~~~~^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_construct.py", line 413, in eye_array
    return _eye(m, n, k, dtype, format)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_construct.py", line 450, in _eye
    return diags_sparse(data, offsets=[k], shape=(m, n), dtype=dtype).asformat(format)
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/_construct.py", line 215, in diags_array
    raise ValueError(f"Offset {offset} (index {j}) out of bounds")
ValueError: Offset 2 (index 0) out of bounds
Falsifying example: test_eye_matches_dense(
    n=1,
    k=2,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.sparse as sp

# Test case 1: n=1, k=2 (basic failing case)
print("Test case 1: n=1, k=2")
print("-" * 40)
np_result = np.eye(1, k=2)
print(f"numpy.eye(1, k=2) = {np_result}")

try:
    sp_result = sp.eye_array(1, k=2)
    print(f"scipy.sparse.eye_array(1, k=2) = {sp_result.toarray()}")
except ValueError as e:
    print(f"scipy.sparse.eye_array(1, k=2) raised ValueError: {e}")

print("\n" + "=" * 50 + "\n")

# Test case 2: n=2, k=3 (another failing case)
print("Test case 2: n=2, k=3")
print("-" * 40)
np_result = np.eye(2, k=3)
print(f"numpy.eye(2, k=3) = \n{np_result}")

try:
    sp_result = sp.eye_array(2, k=3)
    print(f"scipy.sparse.eye_array(2, k=3) = \n{sp_result.toarray()}")
except ValueError as e:
    print(f"scipy.sparse.eye_array(2, k=3) raised ValueError: {e}")

print("\n" + "=" * 50 + "\n")

# Test case 3: n=2, k=-3 (negative diagonal out of bounds)
print("Test case 3: n=2, k=-3")
print("-" * 40)
np_result = np.eye(2, k=-3)
print(f"numpy.eye(2, k=-3) = \n{np_result}")

try:
    sp_result = sp.eye_array(2, k=-3)
    print(f"scipy.sparse.eye_array(2, k=-3) = \n{sp_result.toarray()}")
except ValueError as e:
    print(f"scipy.sparse.eye_array(2, k=-3) raised ValueError: {e}")

print("\n" + "=" * 50 + "\n")

# Test case 4: rectangular matrix (m=1, n=3, k=-2)
print("Test case 4: m=1, n=3, k=-2")
print("-" * 40)
np_result = np.eye(1, 3, k=-2)
print(f"numpy.eye(1, 3, k=-2) = {np_result}")

try:
    sp_result = sp.eye_array(1, 3, k=-2)
    print(f"scipy.sparse.eye_array(1, 3, k=-2) = {sp_result.toarray()}")
except ValueError as e:
    print(f"scipy.sparse.eye_array(1, 3, k=-2) raised ValueError: {e}")

print("\n" + "=" * 50 + "\n")

# Test case 5: working case for comparison (n=3, k=1)
print("Test case 5: n=3, k=1 (working case for comparison)")
print("-" * 40)
np_result = np.eye(3, k=1)
print(f"numpy.eye(3, k=1) = \n{np_result}")

try:
    sp_result = sp.eye_array(3, k=1)
    print(f"scipy.sparse.eye_array(3, k=1) = \n{sp_result.toarray()}")
except ValueError as e:
    print(f"scipy.sparse.eye_array(3, k=1) raised ValueError: {e}")
```

<details>

<summary>
scipy.sparse.eye_array raises ValueError while numpy.eye returns zero matrix
</summary>
```
Test case 1: n=1, k=2
----------------------------------------
numpy.eye(1, k=2) = [[0.]]
scipy.sparse.eye_array(1, k=2) raised ValueError: Offset 2 (index 0) out of bounds

==================================================

Test case 2: n=2, k=3
----------------------------------------
numpy.eye(2, k=3) =
[[0. 0.]
 [0. 0.]]
scipy.sparse.eye_array(2, k=3) raised ValueError: Offset 3 (index 0) out of bounds

==================================================

Test case 3: n=2, k=-3
----------------------------------------
numpy.eye(2, k=-3) =
[[0. 0.]
 [0. 0.]]
scipy.sparse.eye_array(2, k=-3) raised ValueError: Offset -3 (index 0) out of bounds

==================================================

Test case 4: m=1, n=3, k=-2
----------------------------------------
numpy.eye(1, 3, k=-2) = [[0. 0. 0.]]
scipy.sparse.eye_array(1, 3, k=-2) raised ValueError: Offset -2 (index 0) out of bounds

==================================================

Test case 5: n=3, k=1 (working case for comparison)
----------------------------------------
numpy.eye(3, k=1) =
[[0. 1. 0.]
 [0. 0. 1.]
 [0. 0. 0.]]
scipy.sparse.eye_array(3, k=1) =
[[0. 1. 0.]
 [0. 0. 1.]
 [0. 0. 0.]]
```
</details>

## Why This Is A Bug

1. **API Contract Violation**: The `scipy.sparse.eye_array` function is documented as the sparse equivalent of `numpy.eye`. The docstring states it returns "a sparse array with ones on diagonal" and "sparse array (m x n) where the kth diagonal is all ones and everything else is zeros." This implies it should handle all valid input combinations that `numpy.eye` accepts, not throw exceptions for valid parameters.

2. **Behavioral Inconsistency**: When the requested diagonal falls completely outside the matrix bounds, `numpy.eye` sensibly returns a zero matrix of the requested shape. This is logical since there are no positions for the diagonal elements. `scipy.sparse.eye_array` instead raises a `ValueError`, breaking compatibility.

3. **Undocumented Exception**: The documentation for `scipy.sparse.eye_array` does not mention any `ValueError` exceptions or restrictions on the `k` parameter beyond what `numpy.eye` accepts. Users migrating from dense to sparse arrays would not expect this difference.

4. **Migration Barrier**: Code that works correctly with `numpy.eye` will break when switching to `scipy.sparse.eye_array` for the same valid inputs. This violates the principle of least surprise and creates unnecessary friction for users optimizing their code with sparse arrays.

5. **Pattern of Failures**: The bug consistently occurs when:
   - For square matrices (n×n): `k >= n` or `k <= -n`
   - For rectangular matrices (m×n): When the diagonal offset places all diagonal elements outside the matrix bounds

## Relevant Context

The error originates in `/scipy/sparse/_construct.py` at line 215 within the `diags_array` function, which is called by `_eye` at line 450. The `_eye` function computes the diagonal data but doesn't handle the case where the diagonal is completely out of bounds.

The problematic code path:
1. `eye_array` (line 413) calls `_eye(m, n, k, dtype, format)`
2. `_eye` (line 450) calls `diags_sparse(data, offsets=[k], shape=(m, n), dtype=dtype)`
3. `diags_array` (line 215) raises `ValueError` when offset is out of bounds

The issue is that `_eye` computes `data = np.ones((1, max(0, min(m + k, n))), dtype=dtype)` at line 449, which correctly handles out-of-bounds diagonals by creating an empty data array when `min(m + k, n) <= 0`. However, `diags_array` still validates and rejects these offsets.

Documentation links:
- scipy.sparse.eye_array: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.eye_array.html
- numpy.eye: https://numpy.org/doc/stable/reference/generated/numpy.eye.html

## Proposed Fix

The fix should check for out-of-bounds diagonals in `_eye` before calling `diags_sparse` and return an appropriate zero sparse array:

```diff
--- a/scipy/sparse/_construct.py
+++ b/scipy/sparse/_construct.py
@@ -446,6 +446,13 @@ def _eye(m, n, k, dtype, format, as_sparray=True):
             data = np.ones(n, dtype=dtype)
             return coo_sparse((data, (row, col)), (n, n))

+    # Handle out-of-bounds diagonals - return zero sparse array
+    # to match numpy.eye behavior
+    if k >= n or k <= -m:
+        # Return empty sparse array of appropriate shape
+        empty_data = np.array([], dtype=dtype)
+        return coo_sparse((empty_data, (np.array([], dtype=np.int32),
+                          np.array([], dtype=np.int32))), shape=(m, n)).asformat(format)
+
     data = np.ones((1, max(0, min(m + k, n))), dtype=dtype)
     return diags_sparse(data, offsets=[k], shape=(m, n), dtype=dtype).asformat(format)
```