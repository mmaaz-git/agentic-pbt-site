# Bug Report: scipy.signal.windows.get_window Kaiser-Bessel Derived Naming Inconsistency

**Target**: `scipy.signal.windows.get_window`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `get_window` function cannot dispatch to `kaiser_bessel_derived` using the function's actual name (with underscores). Instead, it requires using 'kaiser bessel derived' (with spaces), which violates the principle of least surprise and is inconsistent with how all other window functions work.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.signal import windows
import numpy as np

@given(st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False))
def test_get_window_recognizes_function_names(beta):
    w_direct = windows.kaiser_bessel_derived(10, beta=beta, sym=True)

    w_from_get_window = windows.get_window(('kaiser_bessel_derived', beta), 10, fftbins=False)

    assert np.allclose(w_direct, w_from_get_window), \
        "get_window should accept the actual function name 'kaiser_bessel_derived'"
```

**Failing input**: `beta=8.6` (any value fails)

## Reproducing the Bug

```python
from scipy.signal import windows

result = windows.get_window(('kaiser_bessel_derived', 8.6), 10, fftbins=False)
```

Output:
```
ValueError: Unknown window type.
```

However, using spaces instead of underscores works:
```python
from scipy.signal import windows

result = windows.get_window(('kaiser bessel derived', 8.6), 10, fftbins=False)
```

This succeeds and returns a valid window.

## Why This Is A Bug

1. **Inconsistency**: The function is named `kaiser_bessel_derived` (with underscores), but `get_window` only recognizes `'kaiser bessel derived'` (with spaces).

2. **Violates user expectations**: Users naturally try to use the function name they see in documentation and autocomplete (`kaiser_bessel_derived`), but this fails.

3. **Inconsistent with other windows**: All other parameterized window functions can be called with their exact function names:
   - `get_window(('kaiser', 8.6), 10)` ✓ works
   - `get_window(('gaussian', 0.5), 10)` ✓ works
   - `get_window(('kaiser_bessel_derived', 8.6), 10)` ✗ fails

4. **Poor error message**: The error "Unknown window type" is confusing when the window type clearly exists.

## Fix

Add the underscore version as an alias in the window name mapping:

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -2345,7 +2345,7 @@ _win_equiv_raw = {
     ('general_hamming', ): (general_hamming, True),
     ('hamming', 'hamm', 'ham'): (hamming, False),
     ('hann', ): (hann, False),
-    ('kaiser bessel derived', 'kbd'): (kaiser_bessel_derived, True),
+    ('kaiser bessel derived', 'kaiser_bessel_derived', 'kbd'): (kaiser_bessel_derived, True),
     ('kaiser', 'ksr'): (kaiser, True),
     ('lanczos', 'sinc'): (lanczos, False),
     ('nuttall', 'nutl', 'nut'): (nuttall, False),