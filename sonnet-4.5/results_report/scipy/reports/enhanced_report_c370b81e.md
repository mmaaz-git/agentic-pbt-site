# Bug Report: scipy.signal.windows.tukey NaN Values for Subnormal Alpha

**Target**: `scipy.signal.windows.tukey`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `tukey` window function produces NaN values when the `alpha` parameter is a subnormal positive float (e.g., 5e-324), due to numerical overflow in division operations that produce infinity, leading to NaN after cosine computation.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st, example
from scipy.signal import windows

@given(M=st.integers(min_value=2, max_value=500),
       alpha=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
@example(M=2, alpha=5e-324)  # Specific failing case
@settings(max_examples=200)
def test_tukey_nonnegative(M, alpha):
    w = windows.tukey(M, alpha)
    assert np.all(w >= 0), f"Tukey window should be non-negative for M={M}, alpha={alpha}"
    assert not np.any(np.isnan(w)), f"Tukey window should not contain NaN values for M={M}, alpha={alpha}"
    assert not np.any(np.isinf(w)), f"Tukey window should not contain infinite values for M={M}, alpha={alpha}"

if __name__ == "__main__":
    print("Running Hypothesis tests...")
    test_tukey_nonnegative()
```

<details>

<summary>
**Failing input**: `M=2, alpha=5e-324`
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:946: RuntimeWarning: overflow encountered in divide
  w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:946: RuntimeWarning: invalid value encountered in add
  w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
Running Hypothesis tests...
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 17, in <module>
    test_tukey_nonnegative()
    ~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 6, in test_tukey_nonnegative
    alpha=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 11, in test_tukey_nonnegative
    assert np.all(w >= 0), f"Tukey window should be non-negative for M={M}, alpha={alpha}"
           ~~~~~~^^^^^^^^
AssertionError: Tukey window should be non-negative for M=2, alpha=5e-324
Falsifying explicit example: test_tukey_nonnegative(
    M=2,
    alpha=5e-324,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.signal import windows

M = 2
alpha = 5e-324

w = windows.tukey(M, alpha)
print(f"tukey({M}, {alpha}) = {w}")
print(f"Contains NaN: {np.any(np.isnan(w))}")
print(f"Window values: {w}")
print(f"Window dtype: {w.dtype}")
```

<details>

<summary>
Output shows NaN value in window array
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:946: RuntimeWarning: overflow encountered in divide
  w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:946: RuntimeWarning: invalid value encountered in add
  w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
tukey(2, 5e-324) = [ 0. nan]
Contains NaN: True
Window values: [ 0. nan]
Window dtype: float64
```
</details>

## Why This Is A Bug

The Tukey window function should produce valid window coefficients for all alpha values in the range [0, 1] as documented. The docstring states that `alpha` is "the fraction of the window inside the cosine tapered region" with special cases:
- "If zero, the Tukey window is equivalent to a rectangular window"
- "If one, the Tukey window is equivalent to a Hann window"

The code correctly handles `alpha <= 0` (line 931) by returning a rectangular window, but fails when alpha is a positive subnormal float like `5e-324`. The problematic computation occurs at line 946:

```python
w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
```

When `alpha = 5e-324`, the division `-2.0/alpha` overflows to negative infinity. Computing `cos(infinity)` returns NaN, which propagates through the calculation, violating the mathematical contract that window values should be between 0 and 1.

## Relevant Context

The value `5e-324` is a subnormal (denormalized) floating-point number - the smallest positive float Python can represent. While such extreme values are rare in practice, scientific computing libraries like SciPy should handle numerical edge cases gracefully.

The mathematical definition of the Tukey window (from Wikipedia and signal processing literature) is continuous and well-defined for all alpha >= 0. As alpha approaches 0, the window should smoothly approach a rectangular window (all ones).

Runtime warnings clearly indicate the numerical issue:
- "overflow encountered in divide" - Division by tiny alpha overflows
- "invalid value encountered in add" - NaN propagation from cos(infinity)

Code location: `/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:946`

## Proposed Fix

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -928,7 +928,10 @@ def tukey(M, alpha=0.5, sym=True, *, xp=None, device=None):
     if _len_guards(M):
         return xp.ones(M, dtype=xp.float64, device=device)

-    if alpha <= 0:
+    # Treat very small alpha as 0 to avoid numerical overflow in division
+    # The threshold 1e-10 is chosen to be well above machine epsilon but
+    # small enough that the window is effectively rectangular
+    if alpha <= 1e-10:
         return xp.ones(M, dtype=xp.float64, device=device)
     elif alpha >= 1.0:
         return hann(M, sym=sym, xp=xp, device=device)
```