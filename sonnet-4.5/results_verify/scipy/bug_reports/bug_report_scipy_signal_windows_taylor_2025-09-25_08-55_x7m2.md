# Bug Report: scipy.signal.windows.taylor Missing Input Validation

**Target**: `scipy.signal.windows.taylor`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `taylor` window function silently returns NaN values when given negative `sll` parameter values, despite documentation stating `sll` "should be a positive number." The function lacks input validation and produces invalid output instead of raising an appropriate error.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
import scipy.signal.windows as w

@given(
    M=st.integers(min_value=4, max_value=200),
    nbar=st.integers(min_value=1, max_value=20),
    sll=st.floats(min_value=-100.0, max_value=-10.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_taylor_symmetry(M, nbar, sll):
    assume(nbar < M)
    window = w.taylor(M, nbar=nbar, sll=sll, sym=True)
    assert np.allclose(window, window[::-1], rtol=1e-10)
```

**Failing input**: `M=4, nbar=2, sll=-10.0`

## Reproducing the Bug

```python
import numpy as np
import scipy.signal.windows as w

window = w.taylor(4, nbar=2, sll=-10.0, sym=True)
print(f"Result: {window}")
print(f"Contains NaN: {np.any(np.isnan(window))}")

window_positive = w.taylor(4, nbar=2, sll=10.0, sym=True)
print(f"\nWith positive sll: {window_positive}")
print(f"Contains NaN: {np.any(np.isnan(window_positive))}")
```

Output:
```
Result: [nan nan nan nan]
Contains NaN: True

With positive sll: [1.19950222 1.03422917 1.03422917 1.19950222]
Contains NaN: False
```

## Why This Is A Bug

The documentation explicitly states that `sll` "should be a positive number," but the function:
1. Does not validate this precondition
2. Silently produces NaN values when given negative inputs
3. Emits only a RuntimeWarning about "invalid value encountered in arccosh"

This violates the principle of fail-fast error handling. Users might reasonably interpret `sll` as a negative dB value (e.g., "-30 dB sidelobe suppression") rather than a positive magnitude, leading to silent failures that are difficult to debug.

The function should validate inputs and raise a `ValueError` with a clear message when `sll <= 0`.

## Fix

Add input validation at the beginning of the `taylor` function:

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -1880,6 +1880,9 @@ def taylor(M, nbar=4, sll=30, norm=True, sym=True, *, xp=None, device=None):
     if _len_guards(M):
         return xp.ones(M, dtype=xp.float64, device=device)

+    if sll <= 0:
+        raise ValueError(f"sll must be a positive number, got {sll}")
+
     M, needs_trunc = _extend(M, sym)

     # Original author: David Dunleavy