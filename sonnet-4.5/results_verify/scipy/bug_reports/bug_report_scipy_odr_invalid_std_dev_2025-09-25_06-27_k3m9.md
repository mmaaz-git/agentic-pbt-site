# Bug Report: scipy.odr.RealData Invalid Standard Deviation Handling

**Target**: `scipy.odr.RealData._sd2wt`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.odr.RealData` does not validate that standard deviations (sx, sy) are positive, leading to two issues: (1) zero standard deviations cause division by zero, producing infinite weights, and (2) negative standard deviations are silently accepted and squared, producing logically incorrect positive weights.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst
import numpy as np
from scipy.odr import RealData

@given(
    x=npst.arrays(
        dtype=np.float64,
        shape=st.just(5),
        elements=st.floats(min_value=0.1, max_value=100,
                          allow_nan=False, allow_infinity=False)
    ),
    y=npst.arrays(
        dtype=np.float64,
        shape=st.just(5),
        elements=st.floats(min_value=0.1, max_value=100,
                          allow_nan=False, allow_infinity=False)
    ),
    sy=npst.arrays(
        dtype=np.float64,
        shape=st.just(5),
        elements=st.floats(min_value=-10, max_value=10,
                          allow_nan=False, allow_infinity=False)
    )
)
@settings(max_examples=100)
def test_realdata_invalid_std_dev(x, y, sy):
    rd = RealData(x, y, sy=sy)
    we = rd.we

    for i in range(len(sy)):
        if sy[i] == 0:
            assert np.isinf(we[i]), "Zero std dev causes inf weight"
        if sy[i] < 0:
            assert we[i] > 0, "Negative std dev produces positive weight"
```

**Failing inputs**:
- Zero standard deviation: `sy = [0.1, 0.0, 0.1]` → `we[1] = inf`
- Negative standard deviation: `sy = [0.1, -0.5, 0.1]` → `we[1] = 4.0`

## Reproducing the Bug

```python
import numpy as np
from scipy.odr import RealData

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

sy_zero = np.array([0.1, 0.2, 0.0, 0.1, 0.1])
rd_zero = RealData(x, y, sy=sy_zero)
print(f"Zero std dev: sy[2]={sy_zero[2]} -> we[2]={rd_zero.we[2]}")

sy_neg = np.array([0.1, -0.5, 0.1, 0.1, 0.1])
rd_neg = RealData(x, y, sy=sy_neg)
print(f"Negative std dev: sy[1]={sy_neg[1]} -> we[1]={rd_neg[1]}")
```

Output:
```
<stdin>:394: RuntimeWarning: divide by zero encountered in divide
Zero std dev: sy[2]=0.0 -> we[2]=inf
Negative std dev: sy[1]=-0.5 -> we[1]=4.0
```

## Why This Is A Bug

1. **Contract Violation**: The RealData docstring states that `sx` and `sy` are "standard deviations", which by mathematical definition must be non-negative (typically positive). The parameter name and documentation create an expectation of valid standard deviations.

2. **Silent Failure**: The code does not validate inputs, allowing:
   - **Zero std dev** → Division by zero in `_sd2wt` (line 394 of `_odrpack.py`) → `inf` weights → Potential numerical issues in downstream ODRPACK calculations
   - **Negative std dev** → Silently squared to positive values → Logically incorrect but may appear to work

3. **User Impact**: Users who accidentally pass invalid data (e.g., from a buggy measurement system or data processing error) will either get infinite weights or incorrect weights without any warning or error.

4. **Comparison to scipy conventions**: Other scipy functions that take standard deviations (e.g., `scipy.stats` distributions) validate that they are positive.

## Fix

Add validation in `RealData.__init__` to ensure standard deviations are positive:

```diff
--- a/scipy/odr/_odrpack.py
+++ b/scipy/odr/_odrpack.py
@@ -358,6 +358,14 @@ class RealData(Data):
     def __init__(self, x, y=None, sx=None, sy=None, covx=None, covy=None,
                  fix=None, meta=None):
         if (sx is not None) and (covx is not None):
             raise ValueError("cannot set both sx and covx")
         if (sy is not None) and (covy is not None):
             raise ValueError("cannot set both sy and covy")
+
+        if sx is not None and np.any(np.asarray(sx) <= 0):
+            raise ValueError("Standard deviations sx must be positive")
+        if sy is not None and np.any(np.asarray(sy) <= 0):
+            raise ValueError("Standard deviations sy must be positive")

         # Set flags for __getattr__
         self._ga_flags = {}
```

Alternatively, for backward compatibility, the code could issue a warning instead of raising an error.