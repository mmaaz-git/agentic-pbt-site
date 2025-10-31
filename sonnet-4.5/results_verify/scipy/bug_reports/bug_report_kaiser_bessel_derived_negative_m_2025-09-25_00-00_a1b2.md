# Bug Report: kaiser_bessel_derived Returns Empty Array for Negative M Instead of Raising ValueError

**Target**: `scipy.signal.windows.kaiser_bessel_derived`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`kaiser_bessel_derived` returns an empty array for negative M values instead of raising a `ValueError`, violating its documented contract. This behavior is inconsistent with all other window functions in `scipy.signal.windows`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pytest
import scipy.signal.windows as windows

@given(st.integers(max_value=-1))
def test_negative_M_raises_kaiser_bessel_derived(M):
    with pytest.raises(ValueError):
        windows.kaiser_bessel_derived(M, beta=8.6)
```

**Failing input**: `M=-1, beta=8.6`

## Reproducing the Bug

```python
import scipy.signal.windows as windows

w = windows.kaiser_bessel_derived(-1, beta=8.6)
print(f"Result: {w}")
print(f"Shape: {w.shape}")
print(f"Expected: ValueError but got empty array instead")
```

Output:
```
Result: []
Shape: (0,)
Expected: ValueError but got empty array instead
```

## Why This Is A Bug

1. **Docstring violation**: The function's docstring explicitly states "An exception is thrown when it is negative" (line 8 of the docstring).

2. **Inconsistent behavior**: All 25 other window functions in `scipy.signal.windows` raise `ValueError` for negative M:
   - `hann(-1)` → ValueError
   - `hamming(-1)` → ValueError
   - `blackman(-1)` → ValueError
   - etc.

3. **API contract violation**: Users relying on the documented behavior may not handle the unexpected empty array return.

## Fix

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -1367,7 +1367,11 @@ def kaiser_bessel_derived(M, beta, *, sym=True, xp=None, device=None):
     if not sym:
         raise ValueError(
             "Kaiser-Bessel Derived windows are only defined for symmetric "
             "shapes"
         )
-    elif M < 1:
+    elif M < 0:
+        raise ValueError(
+            "Window length M must be a non-negative integer"
+        )
+    elif M == 0:
         return xp.asarray([])
     elif M % 2:
         raise ValueError(
```