# Bug Report: numpy.fft.rfftfreq Accepts Negative n Values

**Target**: `numpy.fft.rfftfreq`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`numpy.fft.rfftfreq` silently accepts negative `n` values and returns an empty array, while the related function `fftfreq` correctly raises ValueError for negative n.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy.fft as fft

@given(st.integers(min_value=-10, max_value=10))
def test_rfftfreq_zero_or_negative_n(n):
    if n <= 0:
        try:
            result = fft.rfftfreq(n)
            assert False, f"rfftfreq should reject n={n}"
        except (ValueError, ZeroDivisionError):
            pass
    else:
        result = fft.rfftfreq(n)
        assert len(result) == n // 2 + 1
```

**Failing input**: `n=-1` (any negative value fails)

## Reproducing the Bug

```python
import numpy.fft as fft

result = fft.rfftfreq(-1)
print(result)

fft.fftfreq(-1)
```

Output:
```
[]
ValueError: negative dimensions are not allowed
```

## Why This Is A Bug

The behavior is inconsistent with related functions:
1. `fftfreq(-1)` raises `ValueError: negative dimensions are not allowed`
2. `rfft(x, n=-1)` raises `ValueError: Invalid number of FFT data points (-1) specified`
3. `rfftfreq(-1)` silently returns `[]`

Since `rfftfreq` is meant to generate frequency values for use with `rfft`, it should validate inputs consistently and reject negative n values like the other functions do. Silently accepting invalid input can lead to subtle bugs where users don't realize they've passed an invalid parameter.

## Fix

In `_helper.py` where `rfftfreq` is implemented, add input validation:

```diff
--- a/numpy/fft/_helper.py
+++ b/numpy/fft/_helper.py
@@ -xxx,6 +xxx,8 @@ def rfftfreq(n, d=1.0, device=None):
     ...
     """
+    if n < 1:
+        raise ValueError(f"Invalid number of data points ({n}) specified.")
     val = 1.0 / (n * d)
     N = n // 2 + 1
```