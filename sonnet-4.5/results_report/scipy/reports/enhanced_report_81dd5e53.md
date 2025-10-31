# Bug Report: scipy.signal.windows.flattop Normalization Contract Violation

**Target**: `scipy.signal.windows.flattop`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `flattop` window function violates its documented normalization contract by returning a maximum value of 1.000000003 (exceeding 1.0 by 3×10⁻⁹) for all odd values of M greater than 1.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
import scipy.signal.windows as windows


@given(st.integers(min_value=1, max_value=1000))
@settings(max_examples=300)
def test_normalization_property(M):
    w = windows.flattop(M)
    max_val = np.max(w)

    assert max_val <= 1.0, f"flattop({M}) has max value {max_val} > 1.0"

if __name__ == "__main__":
    test_normalization_property()
```

<details>

<summary>
**Failing input**: `M=3`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 15, in <module>
    test_normalization_property()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 7, in test_normalization_property
    @settings(max_examples=300)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 12, in test_normalization_property
    assert max_val <= 1.0, f"flattop({M}) has max value {max_val} > 1.0"
           ^^^^^^^^^^^^^^
AssertionError: flattop(3) has max value 1.000000003 > 1.0
Falsifying example: test_normalization_property(
    M=3,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.signal.windows as windows

# Test odd values of M
print("Testing odd values of M:")
for M in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
    w = windows.flattop(M)
    max_val = np.max(w)
    print(f"flattop({M:2}): max = {max_val:.15f}, exceeds 1.0: {max_val > 1.0}")

print("\nTesting even values of M:")
for M in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
    w = windows.flattop(M)
    max_val = np.max(w)
    print(f"flattop({M:2}): max = {max_val:.15f}, exceeds 1.0: {max_val > 1.0}")

print("\nVerifying coefficient sum:")
coeffs = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
coeff_sum = sum(coeffs)
print(f"Sum of coefficients: {coeff_sum:.15f}")
print(f"Excess over 1.0: {coeff_sum - 1.0:.15e}")
```

<details>

<summary>
Output showing normalization violation for odd M values
</summary>
```
Testing odd values of M:
flattop( 1): max = 1.000000000000000, exceeds 1.0: False
flattop( 3): max = 1.000000003000000, exceeds 1.0: True
flattop( 5): max = 1.000000003000000, exceeds 1.0: True
flattop( 7): max = 1.000000003000000, exceeds 1.0: True
flattop( 9): max = 1.000000003000000, exceeds 1.0: True
flattop(11): max = 1.000000003000000, exceeds 1.0: True
flattop(13): max = 1.000000003000000, exceeds 1.0: True
flattop(15): max = 1.000000003000000, exceeds 1.0: True
flattop(17): max = 1.000000003000000, exceeds 1.0: True
flattop(19): max = 1.000000003000000, exceeds 1.0: True

Testing even values of M:
flattop( 2): max = -0.000421051000000, exceeds 1.0: False
flattop( 4): max = 0.198210530000000, exceeds 1.0: False
flattop( 6): max = 0.606872152576212, exceeds 1.0: False
flattop( 8): max = 0.780873914938770, exceeds 1.0: False
flattop(10): max = 0.862476344072674, exceeds 1.0: False
flattop(12): max = 0.906201246330179, exceeds 1.0: False
flattop(14): max = 0.932114602418717, exceeds 1.0: False
flattop(16): max = 0.948664113986879, exceeds 1.0: False
flattop(18): max = 0.959851200519604, exceeds 1.0: False
flattop(20): max = 0.967756433174765, exceeds 1.0: False

Verifying coefficient sum:
Sum of coefficients: 1.000000003000000
Excess over 1.0: 3.000000026176508e-09
```
</details>

## Why This Is A Bug

The scipy.signal.windows.flattop function documentation explicitly states in its Returns section: "The window, with the maximum value normalized to 1 (though the value 1 does not appear if `M` is even and `sym` is True)." This is an unqualified contract that promises the maximum value will not exceed 1.0.

The implementation violates this contract for all odd values of M > 1 by consistently returning a maximum value of 1.000000003. This occurs because:

1. The function uses five coefficients [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368] that sum to 1.000000003 instead of exactly 1.0
2. The implementation uses these coefficients in a general cosine window via `_general_cosine_impl`
3. For odd M values, when all cosine terms align at the center point (n=(M-1)/2), the window reaches its maximum value equal to the sum of coefficients
4. Since the coefficients sum to 1.000000003, the maximum value is 1.000000003

While the deviation is tiny (3×10⁻⁹), it is a deterministic, systematic violation of the documented behavior rather than random floating-point error. The documentation makes no allowance for tolerance or approximation.

## Relevant Context

The flattop window is specifically designed for "taking accurate measurements of signal amplitude in the frequency domain, with minimal scalloping error" according to the documentation. This precision-oriented use case makes even small normalization errors potentially relevant for users performing strict validation or property-based testing.

The coefficients appear to originate from the referenced source: D'Antona and Ferrero's "Digital Signal Processing for Measurement Systems" (2006), which may explain why they don't sum to exactly 1.0.

Source code location: `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/signal/windows/_windows.py:669-675`

Documentation link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.flattop.html

## Proposed Fix

```diff
--- a/scipy/signal/windows/_windows.py
+++ b/scipy/signal/windows/_windows.py
@@ -669,7 +669,7 @@ def flattop(M, sym=True, *, xp=None, device=None):
     """
     xp = _namespace(xp)
     a = xp.asarray(
-        [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368],
+        [0.21557894737578943, 0.41663157618426313, 0.27726315784210524, 0.08357894726315789, 0.006947367978947367],
         dtype=xp.float64, device=device
     )
     device = xp_device(a)
```