# Bug Report: scipy.cluster.vq - Incorrect Distance with Duplicate Centroids

**Target**: `scipy.cluster.vq.vq`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The optimized C implementation of `vq()` returns incorrect (non-zero) distances when an observation exactly matches a centroid in a codebook that contains duplicate centroids. The Python reference implementation `py_vq()` correctly returns zero distance in the same scenario.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
import numpy as np
import scipy.cluster.vq as vq


@settings(max_examples=200)
@given(
    obs=arrays(dtype=np.float64, shape=st.tuples(st.integers(1, 30), st.integers(1, 10)),
               elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
    code_book=arrays(dtype=np.float64, shape=st.tuples(st.integers(1, 20), st.integers(1, 10)),
                     elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
)
def test_vq_py_vq_equivalence(obs, code_book):
    assume(obs.shape[1] == code_book.shape[1])
    assume(np.all(np.isfinite(obs)))
    assume(np.all(np.isfinite(code_book)))

    code1, dist1 = vq.vq(obs, code_book)
    code2, dist2 = vq.py_vq(obs, code_book)

    assert np.array_equal(code1, code2)
    assert np.allclose(dist1, dist2, rtol=1e-10)
```

**Failing input**: Codebook with duplicate centroids and an observation matching a non-duplicate centroid.

## Reproducing the Bug

```python
import numpy as np
from scipy.cluster.vq import vq, py_vq

obs = np.array([[3.56101766e-04, 3.34689186e+01, 0.0, 0.0, 0.0]])
code_book = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [3.56101766e-04, 3.34689186e+01, 0.0, 0.0, 0.0]
])

code_vq, dist_vq = vq(obs, code_book)
code_py, dist_py = py_vq(obs, code_book)

print(f"vq distance: {dist_vq[0]}")
print(f"py_vq distance: {dist_py[0]}")
print(f"Expected: 0.0 (observation matches centroid 4 exactly)")

assert code_vq[0] == code_py[0] == 4
assert dist_py[0] == 0.0
assert dist_vq[0] != 0.0
```

## Why This Is A Bug

The `vq` function should return the Euclidean distance from each observation to its nearest centroid. When an observation exactly matches a centroid (identical vectors), the Euclidean distance should be exactly 0.0. The Python reference implementation `py_vq` correctly returns 0.0, but the C implementation returns a small non-zero value (approximately 6.7e-07 in this case).

This bug only manifests when:
1. The codebook contains duplicate centroids
2. An observation exactly matches one of the non-duplicate centroids

Without duplicate centroids, `vq` correctly returns 0.0 for exact matches. This suggests a code path in the C implementation that handles duplicate centroids incorrectly, introducing numerical error even for exact matches.

## Fix

The issue is likely in the C implementation's distance calculation or comparison logic when handling duplicate centroids. The fix would involve:

1. Identifying the C code path that handles codebooks with duplicate entries
2. Ensuring exact matches return precisely 0.0 distance, potentially by adding an early equality check before distance calculation
3. Alternatively, ensuring the distance calculation maintains numerical precision for identical vectors

Without access to the C source, a high-level fix would be to add a check in the C code:

```c
if (vectors_are_identical(obs, centroid)) {
    distance = 0.0;
} else {
    distance = compute_euclidean_distance(obs, centroid);
}
```