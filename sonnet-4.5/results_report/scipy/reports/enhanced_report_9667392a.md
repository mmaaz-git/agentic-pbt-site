# Bug Report: scipy.sparse.linalg.spbandwidth Crashes on Empty Sparse Matrices

**Target**: `scipy.sparse.linalg.spbandwidth`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `spbandwidth` function crashes with a `ValueError` when given a sparse matrix that contains no non-zero elements, failing to handle what should be a valid edge case that mathematically has a well-defined bandwidth of (0, 0).

## Property-Based Test

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from hypothesis import given, strategies as st, settings

@given(
    n=st.integers(min_value=1, max_value=20),
    density=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=200, deadline=None)
def test_spbandwidth_no_crash(n, density):
    rng = np.random.RandomState(0)
    A = sp.random(n, n, density=density, format='csr', random_state=rng)

    below, above = spl.spbandwidth(A)

    assert isinstance(below, int) and isinstance(above, int)
    assert 0 <= below < n
    assert 0 <= above < n

if __name__ == "__main__":
    test_spbandwidth_no_crash()
```

<details>

<summary>
**Failing input**: `n=1, density=0.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 22, in <module>
    test_spbandwidth_no_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 7, in test_spbandwidth_no_crash
    n=st.integers(min_value=1, max_value=20),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 15, in test_spbandwidth_no_crash
    below, above = spl.spbandwidth(A)
                   ~~~~~~~~~~~~~~~^^^
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/sparse/linalg/_dsolve/linsolve.py", line 882, in spbandwidth
    return max(-np.min(gap).item(), 0), max(np.max(gap).item(), 0)
                ~~~~~~^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 3301, in min
    return _wrapreduction(a, np.minimum, 'min', axis, None, out,
                          keepdims=keepdims, initial=initial, where=where)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py", line 86, in _wrapreduction
    return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: zero-size array to reduction operation minimum which has no identity
Falsifying example: test_spbandwidth_no_crash(
    n=1,
    density=0.0,
)
```
</details>

## Reproducing the Bug

```python
import scipy.sparse as sp
import scipy.sparse.linalg as spl

# Create a zero sparse matrix (3x3 with no non-zero elements)
A = sp.csr_matrix((3, 3))
print(f"Matrix shape: {A.shape}")
print(f"Number of non-zero elements: {A.nnz}")
print(f"Matrix data: {A.toarray()}")
print()

try:
    below, above = spl.spbandwidth(A)
    print(f"Bandwidth: lower={below}, upper={above}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
Error: ValueError when processing zero matrix
</summary>
```
Matrix shape: (3, 3)
Number of non-zero elements: 0
Matrix data: [[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]

Error: ValueError: zero-size array to reduction operation minimum which has no identity
```
</details>

## Why This Is A Bug

This violates expected behavior in several ways:

1. **Zero matrices are valid sparse matrices**: A sparse matrix with no non-zero elements is a mathematically valid and commonly occurring sparse matrix. The function should handle this edge case gracefully.

2. **Documentation implies all sparse matrices are supported**: The docstring states "Return the lower and upper bandwidth of a 2D numeric array" and "Only the sparse structure is used here. Values are not checked for zeros." There's no mention that matrices must have at least one non-zero element.

3. **Mathematical definition is clear**: For a zero matrix, the bandwidth is mathematically well-defined as (0, 0) since there are no non-zero elements at any distance from the main diagonal.

4. **Function already returns (0, 0) for other cases**: The documentation shows that `spbandwidth(eye_array(3))` returns `(0, 0)`, establishing this as a valid return value. The docstring explicitly states "A zero denotes no sub/super diagonal entries on that side."

5. **Implementation bug, not design choice**: The crash occurs at line 882 when calling `np.min(gap)` on an empty array. For CSR/CSC formats with no non-zero elements, `gap = np.repeat(np.arange(N), np.diff(indptr))` produces an empty array because `np.diff(indptr)` is all zeros when there are no non-zero elements.

## Relevant Context

The bug occurs in `/scipy/sparse/linalg/_dsolve/linsolve.py` at line 882. The issue manifests differently depending on the sparse format:

- For CSR/CSC formats: `gap` becomes an empty array when `np.diff(indptr)` is all zeros
- For COO format: `gap` is empty when `A.coords` has no elements
- For DOK format: The function adds `[0]` to the list, avoiding the crash
- For DIA format: Already handles empty offsets correctly

Other mathematical libraries handle this case properly. For example, MATLAB's bandwidth function documentation explicitly states: "By convention, the upper and lower bandwidths of an empty matrix are both zero."

Documentation link: [scipy.sparse.linalg.spbandwidth](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spbandwidth.html)
Source code: scipy/sparse/linalg/_dsolve/linsolve.py:827-882

## Proposed Fix

```diff
--- a/scipy/sparse/linalg/_dsolve/linsolve.py
+++ b/scipy/sparse/linalg/_dsolve/linsolve.py
@@ -879,4 +879,6 @@ def spbandwidth(A):
     elif A.format == "dok":
         gap = [(c - r) for r, c in A.keys()] + [0]
         return -min(gap), max(gap)
+    if len(gap) == 0:
+        return 0, 0
     return max(-np.min(gap).item(), 0), max(np.max(gap).item(), 0)
```