# Bug Report: scipy.fftpack.next_fast_len accepts 0 despite documentation requiring positive integer

**Target**: `scipy.fftpack.next_fast_len`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.fftpack.next_fast_len(0)` returns 0 instead of raising `ValueError`, violating its documented API contract that requires the target parameter to be a positive integer.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import scipy.fftpack

@given(st.integers(min_value=-1000, max_value=0))
@settings(max_examples=100)
def test_next_fast_len_rejects_non_positive(target):
    try:
        result = scipy.fftpack.next_fast_len(target)
        if target <= 0:
            assert False, f"Should raise ValueError but returned {result}"
    except (ValueError, RuntimeError):
        pass
```

**Failing input**: `target=0`

## Reproducing the Bug

```python
import scipy.fftpack
import numpy as np

result = scipy.fftpack.next_fast_len(0)
print(f"next_fast_len(0) = {result}")

try:
    scipy.fftpack.next_fast_len(-1)
except ValueError as e:
    print(f"next_fast_len(-1) raises: {e}")

try:
    x = np.array([1., 2., 3.])
    scipy.fftpack.fft(x, n=result)
except ValueError as e:
    print(f"Using result in fft: {e}")
```

Output:
```
next_fast_len(0) = 0
next_fast_len(-1) raises: Target length must be positive
Using result in fft: invalid number of data points (0) specified
```

## Why This Is A Bug

1. **Documentation violation**: The docstring explicitly states `target : int - Must be a positive integer`
2. **Inconsistent behavior**: Negative values correctly raise `ValueError: Target length must be positive`, but 0 is silently accepted
3. **Invalid output**: The returned value (0) is not a 5-smooth number and violates the documented postcondition
4. **Unusable result**: Using the returned value in `fft` causes a `ValueError`

## Fix

```diff
--- a/scipy/fftpack/_helper.py
+++ b/scipy/fftpack/_helper.py
@@ -1,5 +1,7 @@
 def next_fast_len(target):
     """..."""
+    if target <= 0:
+        raise ValueError("Target length must be positive")
     return _helper.good_size(target, True)
```