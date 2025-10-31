# Bug Report: scipy.signal.windows.tukey produces NaN with extremely small alpha values

**Target**: `scipy.signal.windows.tukey`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `scipy.signal.windows.tukey` function returns NaN values when called with extremely small alpha parameters (< ~1e-307), causing numerical overflow in the internal division operations.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
import numpy as np
import scipy.signal.windows as windows


@given(st.integers(min_value=1, max_value=100),
       st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@example(2, 2.225e-311)  # Add the specific failing case
@settings(max_examples=1000)
def test_tukey_no_nan(M, alpha):
    """Tukey window should never produce NaN for alpha in [0, 1]."""
    w = windows.tukey(M, alpha)
    assert not np.any(np.isnan(w)), \
        f"tukey({M}, alpha={alpha}) produced NaN: {w}"

if __name__ == "__main__":
    # Run the test
    print("Running Hypothesis test for scipy.signal.windows.tukey")
    print("=" * 60)
    try:
        test_tukey_no_nan()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
```

<details>

<summary>
**Failing input**: `M=2, alpha=2.225e-311`
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:946: RuntimeWarning: overflow encountered in divide
  w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:946: RuntimeWarning: invalid value encountered in add
  w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
Running Hypothesis test for scipy.signal.windows.tukey
============================================================
Test failed: tukey(2, alpha=2.225e-311) produced NaN: [ 0. nan]
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.signal.windows as windows
import warnings

# Show all warnings
warnings.filterwarnings('error')

print("Testing scipy.signal.windows.tukey with extremely small alpha values")
print("=" * 70)

# Test case 1: The specific failing input from hypothesis
print("\nTest 1: Original failing input from hypothesis")
print("-" * 50)
M, alpha = 2, 2.225e-311
try:
    w = windows.tukey(M, alpha)
    print(f"tukey({M}, alpha={alpha:.3e}) = {w}")
    if np.any(np.isnan(w)):
        print("ERROR: NaN values found in output!")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

# Test case 2: Various small alpha values
print("\nTest 2: Various extremely small alpha values")
print("-" * 50)
test_cases = [
    (2, 1e-308),
    (5, 1e-308),
    (10, 1e-308),
    (3, 1e-307),
    (4, 5e-308),
]

for M, alpha in test_cases:
    try:
        warnings.filterwarnings('always')  # Show warnings but don't raise
        with warnings.catch_warnings(record=True) as w_list:
            w = windows.tukey(M, alpha)
            print(f"tukey({M}, alpha={alpha:.2e}) = {w}")
            if np.any(np.isnan(w)):
                print(f"  -> Contains NaN at indices: {np.where(np.isnan(w))[0]}")
            if w_list:
                for warning in w_list:
                    print(f"  -> Warning: {warning.message}")
    except Exception as e:
        print(f"  -> Exception: {type(e).__name__}: {e}")

# Test case 3: Find the threshold where NaN starts appearing
print("\nTest 3: Finding the threshold where NaN appears")
print("-" * 50)
M = 5
alphas = [1e-306, 1e-307, 1e-308, 1e-309, 1e-310]
for alpha in alphas:
    warnings.filterwarnings('always')
    with warnings.catch_warnings(record=True) as w_list:
        w = windows.tukey(M, alpha)
        has_nan = np.any(np.isnan(w))
        print(f"alpha={alpha:.2e}: NaN present = {has_nan}")
        if has_nan:
            print(f"  -> Result: {w}")
```

<details>

<summary>
RuntimeWarning: overflow encountered in divide, NaN values in output
</summary>
```
Testing scipy.signal.windows.tukey with extremely small alpha values
======================================================================

Test 1: Original failing input from hypothesis
--------------------------------------------------
Exception raised: RuntimeWarning: overflow encountered in divide

Test 2: Various extremely small alpha values
--------------------------------------------------
tukey(2, alpha=1.00e-308) = [ 0. nan]
  -> Contains NaN at indices: [1]
  -> Warning: overflow encountered in divide
  -> Warning: invalid value encountered in add
tukey(5, alpha=1.00e-308) = [ 0.  1.  1.  1. nan]
  -> Contains NaN at indices: [4]
  -> Warning: overflow encountered in divide
  -> Warning: invalid value encountered in add
tukey(10, alpha=1.00e-308) = [ 0.  1.  1.  1.  1.  1.  1.  1.  1. nan]
  -> Contains NaN at indices: [9]
  -> Warning: overflow encountered in divide
  -> Warning: invalid value encountered in add
tukey(3, alpha=1.00e-307) = [0. 1. 1.]
tukey(4, alpha=5.00e-308) = [0. 1. 1. 1.]

Test 3: Finding the threshold where NaN appears
--------------------------------------------------
alpha=1.00e-306: NaN present = False
alpha=1.00e-307: NaN present = False
alpha=1.00e-308: NaN present = True
  -> Result: [ 0.  1.  1.  1. nan]
alpha=1.00e-309: NaN present = True
  -> Result: [ 0.  1.  1.  1. nan]
alpha=1.00e-310: NaN present = True
  -> Result: [ 0.  1.  1.  1. nan]
```
</details>

## Why This Is A Bug

This violates the expected behavior of the Tukey window function in several ways:

1. **Undocumented limitation**: The function's docstring states that alpha should be a float between 0 and 1, with special handling for alpha=0 (rectangular window) and alpha=1 (Hann window). There is no documented minimum value above 0, yet the function fails for valid floating-point numbers in the range (0, 1).

2. **Mathematical inconsistency**: The function already handles alpha=0 as a special case returning all ones. As alpha approaches 0, the window should mathematically approach a rectangular window (all ones), but instead it produces NaN values.

3. **Numerical instability**: The implementation at line 946 of `/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py` performs division by alpha:
   ```python
   w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
   ```
   When alpha < ~1e-307, the term `2.0/alpha` overflows to infinity, causing `cos(infinity)` to return NaN.

4. **Inconsistent error handling**: The function validates that M is a positive integer and handles edge cases for alpha (≤0 and ≥1), but fails to handle the numerical overflow case for extremely small positive alpha values.

## Relevant Context

The bug occurs specifically in the calculation of the third segment (`w3`) of the Tukey window, which represents the right taper region. The issue manifests when:
- Alpha values are smaller than approximately 1e-307
- The NaN always appears in the last element(s) of the returned array
- The threshold appears to be related to the floating-point representation limits where `2.0/alpha` exceeds the maximum representable float value (~1.8e308)

The Tukey window is commonly used in signal processing for spectral analysis with controllable edge tapering. While alpha values below 1e-307 are extremely unlikely in practical applications, the function should either:
1. Handle these edge cases gracefully
2. Explicitly document the numerical limitations
3. Raise a clear error for unsupported input ranges

Documentation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html

Code location: scipy/signal/windows/_windows.py, lines 866-950

## Proposed Fix

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -928,11 +928,15 @@ def tukey(M, alpha=0.5, sym=True, *, xp=None, device=None):
     if _len_guards(M):
         return xp.ones(M, dtype=xp.float64, device=device)

     if alpha <= 0:
         return xp.ones(M, dtype=xp.float64, device=device)
     elif alpha >= 1.0:
         return hann(M, sym=sym, xp=xp, device=device)
+    elif alpha < 1e-300:
+        # For extremely small alpha, avoid numerical overflow in division
+        # The window approaches a rectangular window as alpha→0
+        return xp.ones(M, dtype=xp.float64, device=device)

     M, needs_trunc = _extend(M, sym)
```