# Bug Report: scipy.spatial.distance.jensenshannon Division by Zero

**Target**: `scipy.spatial.distance.jensenshannon`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `jensenshannon` distance function returns `nan` when both input probability vectors are all zeros, due to division by zero during normalization.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.spatial.distance import jensenshannon


@given(
    st.lists(st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
)
@settings(max_examples=500)
def test_jensenshannon_identity(p_list):
    p = np.array(p_list)
    d = jensenshannon(p, p)
    assert np.isclose(d, 0.0), f"jensenshannon(p, p) should be 0, got {d}"
```

**Failing input**: `p_list=[0.0, 0.0, 0.0]`

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.distance import jensenshannon

p = np.array([0.0, 0.0, 0.0])
q = np.array([0.0, 0.0, 0.0])
result = jensenshannon(p, q)
assert np.isnan(result)
```

## Why This Is A Bug

1. **Violates identity property**: For any probability vector `p`, `jensenshannon(p, p)` should return 0, but returns `nan` for all-zero vectors.

2. **Violates reasonable expectations**: While all-zero vectors are not valid probability distributions, the function should either handle them gracefully (return 0) or raise a clear error, not silently return `nan`.

3. **Inconsistent with documentation**: The docstring states "This routine will normalize `p` and `q` if they don't sum to 1.0", but normalization fails for all-zero inputs with 0/0 = `nan`.

## Fix

Add a check for all-zero vectors before normalization at lines 1378-1379:

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -1375,8 +1375,16 @@ def jensenshannon(p, q, base=None, *, axis=0, keepdims=False):
     """
     p = np.asarray(p)
     q = np.asarray(q)
-    p = p / np.sum(p, axis=axis, keepdims=True)
-    q = q / np.sum(q, axis=axis, keepdims=True)
+    p_sum = np.sum(p, axis=axis, keepdims=True)
+    q_sum = np.sum(q, axis=axis, keepdims=True)
+
+    if np.any(p_sum == 0) or np.any(q_sum == 0):
+        raise ValueError("Input arrays must have at least one non-zero element")
+
+    p = p / p_sum
+    q = q / q_sum
+
     m = (p + q) / 2.0
     left = rel_entr(p, m)
     right = rel_entr(q, m)
```