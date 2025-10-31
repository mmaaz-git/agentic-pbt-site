# Bug Report: scipy.signal.windows.tukey Returns NaN for Extremely Small Alpha Values

**Target**: `scipy.signal.windows.tukey`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `tukey` window function produces NaN values when called with extremely small but mathematically valid alpha parameter values (alpha < 1e-308), instead of returning a valid window array as expected.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from scipy.signal.windows import tukey


@given(st.integers(min_value=2, max_value=100),
       st.floats(min_value=1e-320, max_value=1e-100, allow_nan=False, allow_infinity=False))
def test_tukey_no_nan_with_tiny_alpha(M, alpha):
    w = tukey(M, alpha=alpha, sym=True)

    assert len(w) == M
    assert np.all(np.isfinite(w)), \
        f"tukey({M}, alpha={alpha}) contains non-finite values: {w}"

if __name__ == "__main__":
    # Run the property test
    test_tukey_no_nan_with_tiny_alpha()
```

<details>

<summary>
**Failing input**: `M=2, alpha=1e-320`
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:946: RuntimeWarning: overflow encountered in divide
  w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:946: RuntimeWarning: invalid value encountered in add
  w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:946: RuntimeWarning: invalid value encountered in cos
  w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 17, in <module>
    test_tukey_no_nan_with_tiny_alpha()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 7, in test_tukey_no_nan_with_tiny_alpha
    st.floats(min_value=1e-320, max_value=1e-100, allow_nan=False, allow_infinity=False))
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 12, in test_tukey_no_nan_with_tiny_alpha
    assert np.all(np.isfinite(w)), \
           ~~~~~~^^^^^^^^^^^^^^^^
AssertionError: tukey(2, alpha=1e-320) contains non-finite values: [ 0. nan]
Falsifying example: test_tukey_no_nan_with_tiny_alpha(
    M=2,  # or any other generated value
    alpha=1e-320,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/54/hypo.py:13
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1085
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.signal.windows import tukey

# Test with extremely small alpha values
print("Testing tukey function with tiny alpha values:")
print("-" * 50)

# Test case 1: M=2, alpha=1e-309
w = tukey(2, alpha=1e-309, sym=True)
print(f"tukey(2, alpha=1e-309) = {w}")
print(f"Contains NaN: {np.any(np.isnan(w))}")
print()

# Test case 2: M=10, alpha=1e-309
w = tukey(10, alpha=1e-309, sym=True)
print(f"tukey(10, alpha=1e-309) = {w}")
print(f"Contains NaN: {np.any(np.isnan(w))}")
print()

# Test case 3: Even smaller alpha - 1e-320
w = tukey(2, alpha=1e-320, sym=True)
print(f"tukey(2, alpha=1e-320) = {w}")
print(f"Contains NaN: {np.any(np.isnan(w))}")
print()

# Test boundary cases to find where it starts failing
print("Finding the boundary where NaN appears:")
print("-" * 50)
test_alphas = [1e-50, 1e-100, 1e-200, 1e-300, 1e-305, 1e-308, 1e-309, 1e-310, 1e-320]
for alpha in test_alphas:
    w = tukey(2, alpha=alpha, sym=True)
    has_nan = np.any(np.isnan(w))
    print(f"alpha={alpha:e}: Contains NaN = {has_nan}")
```

<details>

<summary>
Output demonstrates NaN values appearing for alpha < 1e-308
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:946: RuntimeWarning: overflow encountered in divide
  w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:946: RuntimeWarning: invalid value encountered in add
  w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
Testing tukey function with tiny alpha values:
--------------------------------------------------
tukey(2, alpha=1e-309) = [ 0. nan]
Contains NaN: True

tukey(10, alpha=1e-309) = [ 0.  1.  1.  1.  1.  1.  1.  1.  1. nan]
Contains NaN: True

tukey(2, alpha=1e-320) = [ 0. nan]
Contains NaN: True

Finding the boundary where NaN appears:
--------------------------------------------------
alpha=1.000000e-50: Contains NaN = False
alpha=1.000000e-100: Contains NaN = False
alpha=1.000000e-200: Contains NaN = False
alpha=1.000000e-300: Contains NaN = False
alpha=1.000000e-305: Contains NaN = False
alpha=1.000000e-308: Contains NaN = True
alpha=1.000000e-309: Contains NaN = True
alpha=1.000000e-310: Contains NaN = True
alpha=9.999889e-321: Contains NaN = True
```
</details>

## Why This Is A Bug

The `tukey` function violates its documented behavior and mathematical contract in several ways:

1. **Documentation Contract Violation**: The function's docstring (lines 874-876 in _windows.py) states that `alpha` is "Shape parameter of the Tukey window, representing the fraction of the window inside the cosine tapered region." As a "fraction", valid values are implicitly [0, 1]. Values like 1e-309 are mathematically valid fractions within this range.

2. **Return Value Contract Violation**: The docstring (lines 887-889) promises "The window, with the maximum value normalized to 1" with no mention that NaN values could be returned under any circumstances.

3. **Numerical Overflow**: The root cause is in line 946 of _windows.py:
   ```python
   w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
   ```
   When alpha < 1e-308, the term `-2.0/alpha` overflows to negative infinity, causing `cos(-inf)` to return NaN.

4. **Inconsistent Edge Case Handling**: The function already handles `alpha <= 0` specially (line 931) by returning a rectangular window (all ones). However, it fails to handle the numerical limits at the other extreme of small positive values.

5. **Silent Data Corruption Risk**: NaN values propagate silently through mathematical operations, potentially corrupting entire signal processing pipelines without immediate detection.

## Relevant Context

The bug manifests at the boundary of double-precision floating-point representation. Testing shows:
- Alpha values >= 1e-305 work correctly
- Alpha values <= 1e-308 produce NaN values
- The transition occurs near the limit of what can be represented as 1/alpha without overflow (approximately 1.79e308 for float64)

The function already has precedent for handling edge cases:
- Line 931: `if alpha <= 0:` returns rectangular window
- Line 933: `elif alpha >= 1.0:` returns Hann window

Documentation references:
- Function definition: `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:866`
- Problematic computation: `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:946`

## Proposed Fix

Add a guard to treat extremely small alpha values as alpha=0 (rectangular window), consistent with the existing edge case handling:

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -928,7 +928,8 @@ def tukey(M, alpha=0.5, sym=True, *, xp=None, device=None):
     if _len_guards(M):
         return xp.ones(M, dtype=xp.float64, device=device)

-    if alpha <= 0:
+    # Treat extremely small alpha values as 0 to avoid numerical overflow
+    if alpha <= 0 or alpha < 1e-300:
         return xp.ones(M, dtype=xp.float64, device=device)
     elif alpha >= 1.0:
         return hann(M, sym=sym, xp=xp, device=device)
```