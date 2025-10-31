# Bug Report: scipy.sparse.csr_matrix has_sorted_indices Flag Remains True Despite Unsorted Indices

**Target**: `scipy.sparse.csr_matrix.has_sorted_indices`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `has_sorted_indices` flag in scipy.sparse CSR/CSC matrices incorrectly remains True after direct modification of the indices array, even when the indices are no longer sorted. This creates an inconsistent state where the flag claims indices are sorted but they are demonstrably unsorted.

## Property-Based Test

```python
import numpy as np
import scipy.sparse as sp
from hypothesis import given, strategies as st, settings

@st.composite
def csr_matrices(draw):
    n = draw(st.integers(min_value=2, max_value=20))
    m = draw(st.integers(min_value=2, max_value=20))
    density = draw(st.floats(min_value=0.1, max_value=0.4))
    return sp.random(n, m, density=density, format='csr')

@given(csr_matrices())
@settings(max_examples=50)
def test_sorted_indices_flag_invalidation(A):
    A.sort_indices()
    assert A.has_sorted_indices

    if A.nnz >= 2:
        A.indices[0], A.indices[1] = A.indices[1], A.indices[0]

        indices_actually_sorted = all(
            np.all(A.indices[A.indptr[i]:A.indptr[i+1]][:-1] <=
                   A.indices[A.indptr[i]:A.indptr[i+1]][1:])
            for i in range(A.shape[0]) if A.indptr[i+1] - A.indptr[i] > 1
        )

        if A.has_sorted_indices and not indices_actually_sorted:
            raise AssertionError(f"BUG: has_sorted_indices flag not invalidated after indices modification\n"
                               f"Matrix shape: {A.shape}, nnz: {A.nnz}\n"
                               f"has_sorted_indices: {A.has_sorted_indices}\n"
                               f"Indices after swap: {A.indices[:10]}...")

if __name__ == "__main__":
    test_sorted_indices_flag_invalidation()
```

<details>

<summary>
**Failing input**: `csr_matrix with shape (2, 4) and 2 stored elements`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 34, in <module>
    test_sorted_indices_flag_invalidation()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 13, in test_sorted_indices_flag_invalidation
    @settings(max_examples=50)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 28, in test_sorted_indices_flag_invalidation
    raise AssertionError(f"BUG: has_sorted_indices flag not invalidated after indices modification\n"
    ...<2 lines>...
                       f"Indices after swap: {A.indices[:10]}...")
AssertionError: BUG: has_sorted_indices flag not invalidated after indices modification
Matrix shape: (2, 4), nnz: 2
has_sorted_indices: True
Indices after swap: [3 1]...
Falsifying example: test_sorted_indices_flag_invalidation(
    A=<Compressed Sparse Row sparse matrix of dtype 'float64'
    	with 2 stored elements and shape (2, 4)>,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/11/hypo.py:28
        /home/npc/pbt/agentic-pbt/worker_/11/hypo.py:30
        /home/npc/pbt/agentic-pbt/worker_/11/hypo.py:31
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1304
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1305
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.sparse as sp

# Create a CSR matrix with unsorted indices
data = np.array([1.0, 2.0, 3.0])
indices = np.array([2, 0, 1])
indptr = np.array([0, 3, 3])
A = sp.csr_matrix((data, indices, indptr), shape=(2, 3))

# Sort the indices
A.sort_indices()
print(f"After sort_indices():")
print(f"  has_sorted_indices = {A.has_sorted_indices}")
print(f"  Indices: {A.indices}")

# Verify indices are actually sorted
indices_sorted = np.all(A.indices[:-1] <= A.indices[1:])
print(f"  Indices actually sorted: {indices_sorted}")

# Now swap the first two indices to make them unsorted
print(f"\nSwapping indices[0] and indices[1]...")
A.indices[0], A.indices[1] = A.indices[1], A.indices[0]

# Check the state after modification
print(f"\nAfter direct modification:")
print(f"  has_sorted_indices = {A.has_sorted_indices}")
print(f"  Indices: {A.indices}")

# Verify indices are actually sorted (they shouldn't be)
indices_sorted = np.all(A.indices[:-1] <= A.indices[1:])
print(f"  Indices actually sorted: {indices_sorted}")

# This demonstrates the bug
if A.has_sorted_indices and not indices_sorted:
    print(f"\nBUG DETECTED: has_sorted_indices is {A.has_sorted_indices} but indices are NOT sorted!")
    print("This violates the contract that has_sorted_indices accurately reflects the state.")
```

<details>

<summary>
Bug detected: has_sorted_indices remains True with unsorted indices
</summary>
```
After sort_indices():
  has_sorted_indices = True
  Indices: [0 1 2]
  Indices actually sorted: True

Swapping indices[0] and indices[1]...

After direct modification:
  has_sorted_indices = True
  Indices: [1 0 2]
  Indices actually sorted: False

BUG DETECTED: has_sorted_indices is True but indices are NOT sorted!
This violates the contract that has_sorted_indices accurately reflects the state.
```
</details>

## Why This Is A Bug

This bug violates the fundamental contract of the `has_sorted_indices` attribute. According to the documentation:

1. **Attribute Documentation**: The `has_sorted_indices` property is documented to return "True if the indices of the array/matrix are in sorted order, False otherwise." This creates a clear contract that the flag should accurately reflect the current state of the indices.

2. **Silent Failures**: When `has_sorted_indices` is True but indices are actually unsorted, any algorithm that relies on this flag for optimization or correctness will fail silently. For example:
   - Binary search operations that assume sorted indices will return incorrect results
   - Matrix multiplication optimizations may produce wrong outputs
   - Duplicate detection algorithms may miss duplicates

3. **Public API Violation**: The `indices` attribute is a public, writable NumPy array (not prefixed with underscore). Users can legitimately modify it directly, yet doing so breaks the consistency guarantee of `has_sorted_indices`.

4. **No Documentation Warning**: There is no documentation warning users that modifying the indices array directly will leave the `has_sorted_indices` flag in an incorrect state.

## Relevant Context

The bug affects both CSR (Compressed Sparse Row) and CSC (Compressed Sparse Column) matrix formats in scipy.sparse. For CSR matrices, sorted indices means column indices within each row are in ascending order. For CSC matrices, it means row indices within each column are sorted.

Key observations:
- The `indices` array has `writeable=True` flag, explicitly allowing modifications
- The `sort_indices()` method sets `has_sorted_indices = True` after sorting
- No mechanism exists to detect or prevent direct modifications to the indices array
- The flag is used internally by scipy for performance optimizations

Documentation references:
- [scipy.sparse.csr_matrix documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)
- The indices array is listed as a public attribute in the class documentation

## Proposed Fix

The bug can be fixed by making the indices array read-only after sorting, preventing modifications that would invalidate the flag:

```diff
--- a/scipy/sparse/_compressed.py
+++ b/scipy/sparse/_compressed.py
@@ -1150,6 +1150,8 @@ class _cs_matrix(_data_matrix, _minmax_mixin):
             self.data[:] = self.data[perm]
             self.indices[:] = self.indices[perm]
             self.has_sorted_indices = True
+            # Prevent modifications that would invalidate the sorted flag
+            self.indices.flags.writeable = False

         # Also set read-only when setting the flag directly
         elif hasattr(self, '_has_sorted_indices'):
@@ -1160,6 +1162,15 @@ class _cs_matrix(_data_matrix, _minmax_mixin):
+    def set_shape(self, shape):
+        """When shape changes, make indices writeable again"""
+        super().set_shape(shape)
+        if hasattr(self, 'indices'):
+            self.indices.flags.writeable = True
+            self.has_sorted_indices = False
+
+    # Add a method to explicitly allow modifications when needed
+    def make_indices_writeable(self):
+        """Allow modifications to indices array (will invalidate sorted flag)"""
+        self.indices.flags.writeable = True
+        self.has_sorted_indices = False
```