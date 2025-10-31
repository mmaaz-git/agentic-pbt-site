# Bug Report: scipy.signal.resample Empty Array Crash

**Target**: `scipy.signal.resample`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.signal.resample` crashes with `ValueError` when given an empty array as input, instead of handling the edge case gracefully.

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
def test_resample_empty_and_tiny_arrays(data):
    if len(data) == 0:
        x = np.array([], dtype=np.float64)
        y = signal.resample(x, 10)
        assert len(y) == 10
```

**Failing input**: `data=[]` (empty array)

## Reproducing the Bug

```python
import numpy as np
import scipy.signal as signal

x = np.array([], dtype=np.float64)
y = signal.resample(x, 10)
```

**Output:**
```
ValueError: invalid number of data points (0) specified
```

**Stack trace:**
```
File "scipy/signal/_signaltools.py", line 3775, in resample
    X = sp_fft.rfft(x)
File "scipy/fft/_pocketfft/basic.py", line 58, in r2c
    raise ValueError(f"invalid number of data points ({tmp.shape[axis]}) specified")
```

## Why This Is A Bug

When resampling an empty array to a target length, the function should either:
1. Return an array of the requested length (filled with zeros or appropriate values)
2. Return an empty array with a clear warning
3. Raise a more informative error message

Instead, it crashes deep in the FFT implementation with a cryptic error message. This violates the principle of graceful edge case handling and makes debugging difficult for users.

Similar to the previously reported empty array bug in `scipy.signal.convolve`, this represents inconsistent edge case handling across the scipy.signal module.

## Fix

Add input validation at the beginning of the `resample` function:

```diff
diff --git a/scipy/signal/_signaltools.py b/scipy/signal/_signaltools.py
--- a/scipy/signal/_signaltools.py
+++ b/scipy/signal/_signaltools.py
@@ -3770,6 +3770,10 @@ def resample(x, num, t=None, axis=0, window=None, domain='time'):
     x = np.asarray(x)
     X_ndim = x.ndim

+    # Handle empty array case
+    if x.shape[axis] == 0:
+        raise ValueError("Cannot resample empty array")
+
     # Default window is a Kaiser window with beta=5
     if window is None:
         window = ('kaiser', 5.0)
```

Alternatively, for consistency with mathematical definitions, return an appropriately sized zero array:

```diff
+    # Handle empty array case
+    if x.shape[axis] == 0:
+        new_shape = list(x.shape)
+        new_shape[axis] = num
+        return np.zeros(new_shape, dtype=x.dtype)
```