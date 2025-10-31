# Bug Report: scipy.signal.windows.tukey Symmetry Violation with Small Alpha Values

**Target**: `scipy.signal.windows.tukey`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `tukey` window function produces asymmetric windows when `sym=True` for extremely small alpha values (< 1e-50) due to catastrophic cancellation in floating-point arithmetic.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
import scipy.signal.windows as w

@given(
    M=st.integers(min_value=2, max_value=500),
    alpha=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=300)
def test_tukey_symmetry(M, alpha):
    window = w.tukey(M, alpha, sym=True)
    assert np.allclose(window, window[::-1]), \
        f"tukey({M}, {alpha}, sym=True) is not symmetric"

if __name__ == "__main__":
    test_tukey_symmetry()
```

<details>

<summary>
**Failing input**: `M=2, alpha=1.055865240562731e-79`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 16, in <module>
    test_tukey_symmetry()
    ~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 6, in test_tukey_symmetry
    M=st.integers(min_value=2, max_value=500),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 12, in test_tukey_symmetry
    assert np.allclose(window, window[::-1]), \
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
AssertionError: tukey(2, 1.055865240562731e-79, sym=True) is not symmetric
Falsifying example: test_tukey_symmetry(
    M=2,  # or any other generated value
    alpha=1.055865240562731e-79,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/12/hypo.py:13
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.signal.windows as w

M = 2
alpha = 1e-300

window = w.tukey(M, alpha, sym=True)
print(f"tukey({M}, {alpha}, sym=True) = {window}")
print(f"Reversed: {window[::-1]}")
print(f"Symmetric: {np.allclose(window, window[::-1])}")
print(f"Are values equal? {window[0]} == {window[-1]}: {window[0] == window[-1]}")
```

<details>

<summary>
Asymmetric window output for symmetric parameter
</summary>
```
tukey(2, 1e-300, sym=True) = [0. 1.]
Reversed: [1. 0.]
Symmetric: False
Are values equal? 0.0 == 1.0: False
```
</details>

## Why This Is A Bug

The scipy.signal.windows.tukey documentation explicitly states that when `sym=True`, the function "generates a symmetric window, for use in filter design." This is an unconditional guarantee with no documented exceptions for numerical precision limits.

The bug occurs because of catastrophic cancellation in line 946 of `/scipy/signal/windows/_windows.py`:
```python
w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
```

For very small alpha values (e.g., 1e-300), the expression evaluates incorrectly:
1. `-2.0/alpha` produces approximately -2e+300
2. `2.0*n3/alpha/(M-1)` produces approximately +2e+300 for the last element
3. Python evaluates left-to-right: `(-2e+300 + 1) + 2e+300`
4. The intermediate result `(-2e+300 + 1)` loses the +1 due to floating-point precision limits
5. The final result becomes 0 instead of π, causing `cos(0) = 1` instead of `cos(π) = -1`
6. This yields `w3[0] = 1.0` instead of the correct `0.0`, breaking symmetry

The parameter alpha=1e-300 is within the documented valid range [0, 1], and the documentation makes no mention of precision limitations or exceptions where symmetry might not hold.

## Relevant Context

- The Tukey window (also known as the tapered cosine window) is commonly used in signal processing for spectral analysis and filter design
- The symmetry property is fundamental for filter design applications, as stated in the documentation
- The bug manifests only for alpha values smaller than approximately 1e-50
- For practical signal processing applications, alpha values are typically between 0.1 and 1.0, making this bug unlikely to affect most users
- The mathematical definition of the Tukey window (see Harris 1978, "On the use of windows for harmonic analysis with the discrete Fourier transform") guarantees symmetry when properly computed
- SciPy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html
- Source code location: `/scipy/signal/windows/_windows.py`, lines 859-950

## Proposed Fix

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -943,7 +943,7 @@ def tukey(M, alpha=0.5, sym=True, *, xp=None, device=None):

     w1 = 0.5 * (1 + xp.cos(xp.pi * (-1 + 2.0*n1/alpha/(M-1))))
     w2 = xp.ones(n2.shape, device=device)
-    w3 = 0.5 * (1 + xp.cos(xp.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
+    w3 = 0.5 * (1 + xp.cos(xp.pi * (1 + (-2.0/alpha + 2.0*n3/alpha/(M-1)))))

     w = xp.concat((w1, w2, w3))
```