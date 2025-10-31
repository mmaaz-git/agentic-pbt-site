# Bug Report: scipy.signal.windows.kaiser Returns NaN for Large Beta Values

**Target**: `scipy.signal.windows.kaiser`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `kaiser` window function returns NaN values when beta is large (>= 710) and M is small, due to numerical overflow in the modified Bessel function I₀ calculation, violating the expectation that window functions should always return finite values.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.signal.windows as windows

@given(
    M=st.integers(min_value=2, max_value=20),
    beta=st.floats(min_value=100.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=30)
def test_kaiser_very_large_beta(M, beta):
    window = windows.kaiser(M, beta, sym=True)

    assert len(window) == M
    assert np.all(np.isfinite(window)), \
        f"Kaiser window should have finite values even for large beta"

if __name__ == "__main__":
    test_kaiser_very_large_beta()
```

<details>

<summary>
**Failing input**: `M=3, beta=710.0`
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:1301: RuntimeWarning: invalid value encountered in divide
  w = (special.i0(beta * xp.sqrt(1 - ((n - alpha) / alpha) ** 2.0)) /
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 18, in <module>
    test_kaiser_very_large_beta()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 6, in test_kaiser_very_large_beta
    M=st.integers(min_value=2, max_value=20),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 14, in test_kaiser_very_large_beta
    assert np.all(np.isfinite(window)), \
           ~~~~~~^^^^^^^^^^^^^^^^^^^^^
AssertionError: Kaiser window should have finite values even for large beta
Falsifying example: test_kaiser_very_large_beta(
    M=3,
    beta=710.0,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/1/hypo.py:15
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.signal.windows as windows

M = 3
beta = 710.0
window = windows.kaiser(M, beta)

print(f"windows.kaiser({M}, {beta}) = {window}")
print(f"Contains NaN: {np.any(np.isnan(window))}")
print(f"All finite: {np.all(np.isfinite(window))}")

assert np.all(np.isfinite(window)), "Kaiser window should not contain NaN values"
```

<details>

<summary>
RuntimeWarning and AssertionError due to NaN in window output
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:1301: RuntimeWarning: invalid value encountered in divide
  w = (special.i0(beta * xp.sqrt(1 - ((n - alpha) / alpha) ** 2.0)) /
windows.kaiser(3, 710.0) = [ 0. nan  0.]
Contains NaN: True
All finite: False
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/1/repo.py", line 12, in <module>
    assert np.all(np.isfinite(window)), "Kaiser window should not contain NaN values"
           ~~~~~~^^^^^^^^^^^^^^^^^^^^^
AssertionError: Kaiser window should not contain NaN values
```
</details>

## Why This Is A Bug

The kaiser window function fails to handle numerical overflow in the modified Bessel function I₀, which grows exponentially with its argument. When beta >= 710, `scipy.special.i0(beta)` returns infinity, causing division by infinity to produce NaN values. This violates multiple expectations:

1. **Silent corruption**: The function returns NaN values without raising an exception, allowing invalid values to propagate through scientific computations undetected.

2. **Contradicts documented return value**: The docstring explicitly states at line 1207 that the function returns "The window, with the maximum value normalized to 1", not mentioning the possibility of NaN values in the Returns section.

3. **Inadequate warning**: While the Notes section at lines 1242-1245 mentions "as beta gets large, the window narrows... otherwise NaNs will be returned", this warning:
   - Does not specify what "large" means (turns out to be beta >= 710)
   - Does not specify what M values are "large enough" to avoid NaN
   - Is buried in the Notes section rather than prominently displayed
   - Provides no actionable guidance for users

4. **Inconsistent with scipy conventions**: Most scipy numerical functions either handle edge cases gracefully or raise clear exceptions with actionable error messages.

## Relevant Context

The bug occurs at line 1301 of `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/signal/windows/_windows.py`:

```python
w = (special.i0(beta * xp.sqrt(1 - ((n - alpha) / alpha) ** 2.0)) /
     special.i0(xp.asarray(beta, dtype=xp.float64)))
```

Testing shows the overflow threshold:
- `scipy.special.i0(709.0) = 1.23e+306` (finite)
- `scipy.special.i0(710.0) = inf` (overflow)

The Kaiser window is widely used in signal processing for filter design and spectral analysis. Users running parameter sweeps or optimization routines can easily encounter beta values above 710, especially when exploring trade-offs between main lobe width and side lobe suppression.

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.kaiser.html
Source code: https://github.com/scipy/scipy/blob/main/scipy/signal/windows/_windows.py#L1185-L1304

## Proposed Fix

The function should validate inputs to prevent numerical overflow and provide clear error messages:

```diff
def kaiser(M, beta, sym=True, *, xp=None, device=None):
    xp = _namespace(xp)

    if _len_guards(M):
        return xp.ones(M, dtype=xp.float64, device=device)
+
+   # Validate beta to prevent numerical overflow in Bessel function
+   # scipy.special.i0 overflows to inf at beta >= 710
+   if beta >= 710:
+       raise ValueError(
+           f"beta={beta} is too large and will cause numerical overflow. "
+           f"The modified Bessel function I₀(beta) overflows for beta >= 710. "
+           f"Consider using a smaller beta value (recommended: beta < 700) "
+           f"or increasing M to better sample the narrow window."
+       )

    M, needs_trunc = _extend(M, sym)
    n = xp.arange(0, M, dtype=xp.float64, device=device)
    alpha = (M - 1) / 2.0
    w = (special.i0(beta * xp.sqrt(1 - ((n - alpha) / alpha) ** 2.0)) /
         special.i0(xp.asarray(beta, dtype=xp.float64)))

    return _truncate(w, needs_trunc)
```