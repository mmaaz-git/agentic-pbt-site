# Bug Report: scipy.stats.quantile Rejects Integer p Values

**Target**: `scipy.stats.quantile`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.stats.quantile` raises a `ValueError` when passed integer values for the probability parameter `p`, even though this is a natural and mathematically valid usage. This violates user expectations and creates an inconsistency with similar NumPy functions like `percentile`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from scipy import stats

@given(
    data=st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=100
    )
)
def test_quantile_zero_is_min(data):
    x = np.array(data)
    q0 = stats.quantile(x, 0)
    min_x = np.min(x)
    assert np.allclose(q0, min_x), f"quantile(x, 0) should be min(x)"
```

**Failing input**: `p=0` (Python integer) or `p=1` (Python integer)

## Reproducing the Bug

```python
import numpy as np
from scipy import stats

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

stats.quantile(x, 0)
```

Output:
```
ValueError: `p` must have real floating dtype.
```

However, this works fine:
```python
stats.quantile(x, 0.0)  # Returns: 1.0
```

And NumPy's similar function accepts integers without issue:
```python
np.percentile(x, 0)     # Returns: 1.0
np.percentile(x, 100)   # Returns: 5.0
```

## Why This Is A Bug

1. **Mathematical validity**: The values 0 and 1 are valid probabilities for quantile computation. There is no mathematical reason to reject them based on dtype.

2. **User expectations**: Users naturally expect `quantile(x, 0)` to return the minimum and `quantile(x, 1)` to return the maximum. The integer literals 0 and 1 are the most natural way to express these.

3. **API inconsistency**: NumPy's `percentile` function accepts integer p values without issue, creating an inconsistency in the SciPy/NumPy ecosystem.

4. **Documentation gap**: The documentation states "Values must be between 0 and 1 (inclusive)" but doesn't mention that they must be floating-point type. This is an implementation detail that shouldn't be exposed to users.

5. **Type coercion norm**: Most NumPy/SciPy functions automatically coerce compatible types. Rejecting integers for `p` violates this expectation.

## Fix

The fix should automatically convert integer `p` values to float. In `_quantile_iv`:

```diff
--- a/scipy/stats/_quantile.py
+++ b/scipy/stats/_quantile.py
@@ -13,7 +13,10 @@ def _quantile_iv(x, p, method, axis, nan_policy, keepdims):
         raise ValueError("`x` must have real dtype.")

     if not xp.isdtype(xp.asarray(p).dtype, ('integral', 'real floating')):
-        raise ValueError("`p` must have real floating dtype.")
+        raise ValueError("`p` must have real dtype.")
+
+    # Ensure p is floating point
+    p = xp.asarray(p, dtype=xp.float64)

     x, p = xp_promote(x, p, force_floating=True, xp=xp)
```

Or more simply, change the error check to accept both integral and real floating dtypes, and rely on `xp_promote` with `force_floating=True` to handle the conversion (which happens on line 18 anyway).