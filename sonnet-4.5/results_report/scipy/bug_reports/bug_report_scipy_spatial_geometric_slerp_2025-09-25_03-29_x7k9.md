# Bug Report: scipy.spatial.geometric_slerp Shape Inconsistency

**Target**: `scipy.spatial.geometric_slerp`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`geometric_slerp` returns inconsistent output shapes when called with a scalar `t` parameter: it returns a 2-D array when `start == end`, but a 1-D array when `start != end`.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages')

import numpy as np
from scipy.spatial import geometric_slerp
from hypothesis import given, strategies as st, settings


def unit_vector_strategy(dims):
    def make_unit_vector(coeffs):
        v = np.array(coeffs)
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            v = np.zeros(len(coeffs))
            v[0] = 1.0
            return v
        return v / norm

    return st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e3, max_value=1e3),
        min_size=dims,
        max_size=dims
    ).map(make_unit_vector)


@settings(max_examples=100)
@given(
    unit_vector_strategy(3),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
)
def test_geometric_slerp_scalar_shape_consistency(start, t):
    end_same = start.copy()
    end_different = -start

    result_same = geometric_slerp(start, end_same, t)
    result_different = geometric_slerp(start, end_different, t)

    assert result_same.shape == result_different.shape, \
        f"Shape mismatch: {result_same.shape} vs {result_different.shape}"
```

**Failing input**: `start = np.array([1.0, 0.0, 0.0])`, `end = np.array([1.0, 0.0, 0.0])`, `t = 0.5`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages')

import numpy as np
from scipy.spatial import geometric_slerp

start = np.array([1.0, 0.0, 0.0])
end_same = np.array([1.0, 0.0, 0.0])
end_different = np.array([0.0, 1.0, 0.0])

result_same = geometric_slerp(start, end_same, 0.5)
result_different = geometric_slerp(start, end_different, 0.5)

print(f"When start == end: shape = {result_same.shape}")
print(f"When start != end: shape = {result_different.shape}")
print(f"Expected: Both should have shape (3,)")
```

Output:
```
When start == end: shape = (1, 3)
When start != end: shape = (3,)
Expected: Both should have shape (3,)
```

## Why This Is A Bug

The function's docstring states that "The result may be 1-dimensional if `t` is a float." This promise is violated when `start == end`. The early return on line 198 of `_geometric_slerp.py` returns `np.linspace(start, start, t.size)`, which produces a 2-D array when `start` is a 1-D array, even for scalar `t`. This breaks the API contract and creates inconsistent behavior.

## Fix

```diff
--- a/scipy/spatial/_geometric_slerp.py
+++ b/scipy/spatial/_geometric_slerp.py
@@ -195,7 +195,10 @@ def geometric_slerp(
                          "space")

     if np.array_equal(start, end):
-        return np.linspace(start, start, t.size)
+        if t.ndim == 0:
+            return start.copy()
+        else:
+            return np.tile(start, (t.size, 1))

     # for points that violate equation for n-sphere
     for coord in [start, end]:
```