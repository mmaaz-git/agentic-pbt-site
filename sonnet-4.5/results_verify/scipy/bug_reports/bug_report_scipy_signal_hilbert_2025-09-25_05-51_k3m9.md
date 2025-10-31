# Bug Report: scipy.signal.hilbert Empty Array Crash

**Target**: `scipy.signal.hilbert`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.signal.hilbert` crashes with `ValueError: N must be positive.` when given an empty array as input, instead of handling the edge case gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.signal as signal

@settings(max_examples=100)
@given(
    data=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                  min_size=0, max_size=5)
)
def test_hilbert_empty_and_tiny_arrays(data):
    x = np.array(data, dtype=np.float64)

    if len(x) == 0:
        result = signal.hilbert(x)
        assert len(result) == 0
    else:
        result = signal.hilbert(x)
        assert len(result) == len(x)
```

**Failing input**: `data=[]` (empty array)

## Reproducing the Bug

```python
import numpy as np
import scipy.signal as signal

x = np.array([], dtype=np.float64)
result = signal.hilbert(x)
```

**Output:**
```
ValueError: N must be positive.
```

**Stack trace:**
```
File "scipy/signal/_signaltools.py", line 2578, in hilbert
    raise ValueError("N must be positive.")
```

## Why This Is A Bug

The Hilbert transform of an empty signal should logically return an empty analytic signal. The current implementation explicitly checks for this case and raises an error, but this is unnecessarily restrictive.

Mathematically, the Hilbert transform is well-defined for an empty set (it's simply the empty set), so there's no mathematical reason to reject empty arrays. The error message also lacks actionable information for users encountering this edge case.

This is part of a broader pattern of inconsistent empty array handling in scipy.signal (see also: `convolve`, `resample`).

## Fix

Modify the `hilbert` function to handle empty arrays gracefully:

```diff
diff --git a/scipy/signal/_signaltools.py b/scipy/signal/_signaltools.py
--- a/scipy/signal/_signaltools.py
+++ b/scipy/signal/_signaltools.py
@@ -2570,10 +2570,15 @@ def hilbert(x, N=None, axis=-1):
     x = asarray(x)
     Nx = x.shape[axis]
     if N is None:
         N = Nx
-    if N <= 0:
-        raise ValueError("N must be positive.")
+    if N < 0:
+        raise ValueError("N must be non-negative.")
+
+    # Handle empty array case
+    if Nx == 0 or N == 0:
+        shape = list(x.shape)
+        shape[axis] = N
+        return np.zeros(shape, dtype=complex)

     Xf = sp_fft.fft(x, N, axis=axis)
     h = zeros(N)
```