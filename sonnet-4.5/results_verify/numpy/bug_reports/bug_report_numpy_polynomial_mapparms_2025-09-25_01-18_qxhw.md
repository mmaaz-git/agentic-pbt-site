# Bug Report: numpy.polynomial.polyutils.mapparms Zero Division

**Target**: `numpy.polynomial.polyutils.mapparms`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `mapparms` and `mapdomain` functions crash with `ZeroDivisionError` when given a zero-width domain (where `old[0] == old[1]`).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import numpy.polynomial.polyutils as pu

@settings(max_examples=200)
@given(st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=1, max_size=10))
def test_mapdomain_round_trip(vals):
    old_domain = [min(vals), max(vals)]
    new_domain = [0, 1]

    mapped = pu.mapdomain(vals, old_domain, new_domain)
    recovered = pu.mapdomain(mapped, new_domain, old_domain)

    np.testing.assert_allclose(vals, recovered, rtol=1e-10, atol=1e-10)
```

**Failing input**: `vals=[0.0]`

## Reproducing the Bug

```python
import numpy.polynomial.polyutils as pu

vals = [5.0]
old_domain = [5.0, 5.0]
new_domain = [0, 1]

result = pu.mapdomain(vals, old_domain, new_domain)
```

Output:
```
ZeroDivisionError: float division by zero
  File "numpy/polynomial/polyutils.py", line 356, in mapdomain
    off, scl = mapparms(old, new)
  File "numpy/polynomial/polyutils.py", line 286, in mapparms
    off = (old[1] * new[0] - old[0] * new[1]) / oldlen
```

## Why This Is A Bug

When a domain has zero width (both endpoints are the same), the `mapparms` function computes `oldlen = old[1] - old[0] = 0` and then divides by this value, causing a crash. This can occur legitimately when:

1. Fitting a polynomial to data with a single unique x-value
2. Analyzing constant-valued data
3. Edge cases in domain manipulation

The function should handle this edge case gracefully, either by returning an identity mapping or raising a more informative error.

## Fix

```diff
--- a/numpy/polynomial/polyutils.py
+++ b/numpy/polynomial/polyutils.py
@@ -283,6 +283,9 @@ def mapparms(old, new):
     old = np.array(old, dtype=float, copy=True, ndmin=1)
     new = np.array(new, dtype=float, copy=True, ndmin=1)
     oldlen = old[1] - old[0]
+    if oldlen == 0:
+        raise ValueError("The domain old has zero width (old[0] == old[1]). "
+                         "Cannot map to a new domain.")
     newlen = new[1] - new[0]
     off = (old[1] * new[0] - old[0] * new[1]) / oldlen
     scl = newlen / oldlen
```
