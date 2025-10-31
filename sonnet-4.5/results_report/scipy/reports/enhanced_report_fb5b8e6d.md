# Bug Report: scipy.signal.windows.tukey Produces NaN with Very Small Alpha Values

**Target**: `scipy.signal.windows.tukey`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `tukey` window function produces NaN values when the `alpha` parameter is extremely small but non-zero (approximately 5e-324 to 1e-310), due to numerical overflow in the computation `-2.0/alpha` which exceeds float64 maximum representable value.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import scipy.signal.windows as windows
import numpy as np

@settings(max_examples=300)
@given(
    M=st.integers(min_value=3, max_value=1000),
    alpha=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False)
)
def test_tukey_no_nan(M, alpha):
    result = windows.tukey(M, alpha=alpha, sym=True)
    assert not np.any(np.isnan(result)), f"tukey({M}, alpha={alpha}) contains NaN"
    assert not np.any(np.isinf(result)), f"tukey({M}, alpha={alpha}) contains inf"

if __name__ == "__main__":
    test_tukey_no_nan()
```

<details>

<summary>
**Failing input**: `M=3, alpha=5e-324`
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:946: RuntimeWarning: overflow encountered in divide
  w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:946: RuntimeWarning: invalid value encountered in add
  w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 16, in <module>
    test_tukey_no_nan()
    ~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 6, in test_tukey_no_nan
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 12, in test_tukey_no_nan
    assert not np.any(np.isnan(result)), f"tukey({M}, alpha={alpha}) contains NaN"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: tukey(3, alpha=5e-324) contains NaN
Falsifying example: test_tukey_no_nan(
    M=3,
    alpha=5e-324,
)
```
</details>

## Reproducing the Bug

```python
import scipy.signal.windows as w
import numpy as np

# Test case that demonstrates the NaN issue
result = w.tukey(3, alpha=1e-313, sym=True)
print(f'tukey(3, alpha=1e-313) = {result}')
print(f'Contains NaN: {np.any(np.isnan(result))}')
print()

# Test multiple alpha values to show the boundary
print("Testing different alpha values:")
for alpha in [1e-320, 1e-313, 1e-310, 1e-300, 1e-10]:
    result = w.tukey(5, alpha=alpha)
    has_nan = np.any(np.isnan(result))
    print(f'alpha={alpha:.2e}: has_nan={has_nan}, result={result}')
```

<details>

<summary>
NaN values appear for alpha between 5e-324 and 1e-310
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:946: RuntimeWarning: overflow encountered in divide
  w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:946: RuntimeWarning: invalid value encountered in add
  w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
tukey(3, alpha=1e-313) = [ 0.  1. nan]
Contains NaN: True

Testing different alpha values:
alpha=1.00e-320: has_nan=True, result=[ 0.  1.  1.  1. nan]
alpha=1.00e-313: has_nan=True, result=[ 0.  1.  1.  1. nan]
alpha=1.00e-310: has_nan=True, result=[ 0.  1.  1.  1. nan]
alpha=1.00e-300: has_nan=False, result=[0. 1. 1. 1. 1.]
alpha=1.00e-10: has_nan=False, result=[0. 1. 1. 1. 0.]
```
</details>

## Why This Is A Bug

This violates the expected behavior for multiple reasons:

1. **The function accepts alpha in the range [0, 1]** according to the documentation, with no mention of limitations on how small alpha can be. The value 5e-324 (the smallest positive float64) is technically within this valid range.

2. **The function promises to return an ndarray with values normalized to max of 1**, but instead returns NaN values which are not valid window coefficients and cannot be used in signal processing operations.

3. **The documentation explicitly states special cases**: alpha=0 returns a rectangular window, and alpha=1 returns a Hann window. By mathematical continuity, very small positive alpha values should behave similarly to alpha=0, not produce NaN.

4. **Runtime warnings reveal the root cause**: The expression `-2.0/alpha` on line 946 causes overflow when alpha < ~1e-308, since 2.0/1e-308 exceeds the maximum float64 value (~1.8e308). This overflow produces infinity, which when passed through `cos()` and addition operations, results in NaN.

5. **The implementation already handles edge cases**: Lines 931-932 check for `alpha <= 0` and return a rectangular window, demonstrating that the function is designed to handle boundary conditions gracefully. The failure to handle very small positive values is an oversight.

## Relevant Context

The bug occurs in `/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py` at line 946:

```python
w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
```

This line computes the trailing taper of the Tukey window. The division by alpha appears multiple times, but the term `-2.0/alpha` is problematic because it's not scaled by any other factor that could keep it within float64 bounds.

The Tukey window is defined mathematically as having three regions:
- Leading taper (computed in w1)
- Flat middle section (w2)
- Trailing taper (computed in w3, where the bug occurs)

For reference, the float64 format has:
- Maximum value: ~1.797693134862315e308
- Minimum positive normal value: ~2.225073858507201e-308
- Minimum positive subnormal value: 5e-324

When alpha approaches 5e-324, the division 2.0/alpha approaches 4e323, which far exceeds float64 max.

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html

## Proposed Fix

The simplest and most robust fix is to treat very small alpha values the same as alpha=0:

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -928,7 +928,9 @@ def tukey(M, alpha=0.5, sym=True, *, xp=None, device=None):
     if _len_guards(M):
         return xp.ones(M, dtype=xp.float64, device=device)

-    if alpha <= 0:
+    # Treat very small alpha as 0 to avoid numerical overflow in division
+    # The threshold 1e-300 is well below any practical alpha value but avoids overflow
+    if alpha <= 0 or alpha < 1e-300:
         return xp.ones(M, dtype=xp.float64, device=device)
     elif alpha >= 1.0:
         return hann(M, sym=sym, xp=xp, device=device)
```