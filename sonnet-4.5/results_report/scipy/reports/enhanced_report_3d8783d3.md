# Bug Report: scipy.signal.deconvolve ValueError with Leading Zero Coefficient

**Target**: `scipy.signal.deconvolve`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`scipy.signal.deconvolve` crashes with ValueError when the divisor array has a leading zero coefficient, even though such polynomials are mathematically valid divisors.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from scipy import signal


def array_1d_strategy():
    return st.lists(
        st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20
    ).map(np.array)


@settings(max_examples=200, deadline=5000)
@given(array_1d_strategy(), array_1d_strategy())
def test_deconvolve_property(signal_arr, divisor):
    assume(len(divisor) > 0 and len(signal_arr) >= len(divisor))
    assume(np.max(np.abs(divisor)) > 0.1)

    quotient, remainder = signal.deconvolve(signal_arr, divisor)

    reconstructed = signal.convolve(divisor, quotient, mode='full') + remainder
    trimmed_reconstructed = reconstructed[:len(signal_arr)]

    assert np.allclose(trimmed_reconstructed, signal_arr, rtol=1e-6, atol=1e-10)

if __name__ == "__main__":
    test_deconvolve_property()
```

<details>

<summary>
**Failing input**: `signal_arr=array([0.0, 0.0]), divisor=array([0.0, 1.0])`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 28, in <module>
  |     test_deconvolve_property()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 15, in test_deconvolve_property
  |     @given(array_1d_strategy(), array_1d_strategy())
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 25, in test_deconvolve_property
    |     assert np.allclose(trimmed_reconstructed, signal_arr, rtol=1e-6, atol=1e-10)
    |            ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError
    | Falsifying example: test_deconvolve_property(
    |     signal_arr=array([1.0, 1.0]),
    |     divisor=array([2.0542927103570503e-28, 1.0]),
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 20, in test_deconvolve_property
    |     quotient, remainder = signal.deconvolve(signal_arr, divisor)
    |                           ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/signal/_signaltools.py", line 2471, in deconvolve
    |     quot = lfilter(num, den, input)
    |   File "/home/npc/.local/lib/python3.13/site-packages/scipy/signal/_signaltools.py", line 2303, in lfilter
    |     result =_sigtools._linear_filter(b, a, x, axis)
    | ValueError: BUG: filter coefficient a[0] == 0 not supported yet
    | Falsifying example: test_deconvolve_property(
    |     signal_arr=array([0.0, 0.0]),
    |     divisor=array([0.0, 1.0]),
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy import signal

signal_arr = np.array([1.0, 2.0, 3.0])
divisor = np.array([0.0, 1.0])

try:
    quotient, remainder = signal.deconvolve(signal_arr, divisor)
    print(f"Success: quotient={quotient}, remainder={remainder}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError: BUG: filter coefficient a[0] == 0 not supported yet
</summary>
```
Error: ValueError: BUG: filter coefficient a[0] == 0 not supported yet

```
</details>

## Why This Is A Bug

The divisor `[0.0, 1.0]` represents the polynomial `0*x + 1 = 1`, which is mathematically equivalent to the constant polynomial 1. This is a valid divisor for polynomial division. The function crashes even though:

1. The documentation states the function should return quotient and remainder such that `signal = convolve(divisor, quotient) + remainder` with no restriction on leading zeros
2. The documentation references `numpy.polydiv` as performing "the same operation", and numpy.polydiv correctly handles polynomials with leading zeros by normalizing them
3. The error message itself admits this is a "BUG" that is "not supported yet" rather than invalid input
4. The function works correctly when the leading zero is manually stripped (e.g., using `[1.0]` instead of `[0.0, 1.0]`)

The crash occurs at line 2471 of `_signaltools.py` when deconvolve calls `lfilter(num, den, input)` with a divisor that has `a[0] == 0`. The underlying C extension `_sigtools._linear_filter` cannot handle this case and raises the error.

## Relevant Context

The deconvolve function is located in `/scipy/signal/_signaltools.py` at lines 2414-2473. The function uses `lfilter` internally for the actual deconvolution computation when the divisor length is <= signal length. The error originates from the C extension module when lfilter is called with a denominator coefficient where `a[0] == 0`.

Testing confirms that manually stripping leading zeros resolves the issue - when using divisor `[1.0]` instead of `[0.0, 1.0]`, the function works correctly and produces the expected result where the original signal can be perfectly reconstructed using the convolution property.

Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.deconvolve.html

## Proposed Fix

```diff
--- a/scipy/signal/_signaltools.py
+++ b/scipy/signal/_signaltools.py
@@ -2455,6 +2455,15 @@ def deconvolve(signal, divisor):
     xp = array_namespace(signal, divisor)

     num = xpx.atleast_nd(xp.asarray(signal), ndim=1, xp=xp)
     den = xpx.atleast_nd(xp.asarray(divisor), ndim=1, xp=xp)
+
+    # Strip leading zeros from divisor to avoid "BUG: filter coefficient a[0] == 0" error
+    if xp.all(den == 0):
+        raise ValueError("divisor must have at least one non-zero element")
+    # Find first non-zero element
+    first_nonzero = int(xp.argmax(xp.abs(den) > 0))
+    if first_nonzero > 0:
+        den = den[first_nonzero:]
+
     if num.ndim > 1:
         raise ValueError("signal must be 1-D.")
     if den.ndim > 1:
```