# Bug Report: scipy.linalg.invpascal Integer Overflow

**Target**: `scipy.linalg.invpascal`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`scipy.linalg.invpascal(n, exact=True)` with `n >= 19` produces incorrect results due to silent integer overflow during matrix multiplication, despite claiming to provide "exact" integer arithmetic.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.linalg

@given(st.integers(min_value=1, max_value=20))
@settings(max_examples=100)
def test_pascal_invpascal_are_inverses(n):
    P = scipy.linalg.pascal(n)
    P_inv = scipy.linalg.invpascal(n)
    product = P @ P_inv
    expected_identity = np.eye(n)
    assert np.allclose(product, expected_identity, rtol=1e-10, atol=1e-10)
```

**Failing input**: `n=19`

## Reproducing the Bug

```python
import numpy as np
import scipy.linalg

n = 19
P = scipy.linalg.pascal(n, exact=True)
P_inv = scipy.linalg.invpascal(n, exact=True)
product = P @ P_inv

print("Product (should be identity):")
print(product)
print(f"\nMax error: {np.max(np.abs(product - np.eye(n)))}")

expected = np.eye(n)
diff = product - expected
rows, cols = np.where(np.abs(diff) > 0.5)
print(f"\n{len(rows)} positions with large errors:")
for r, c in zip(rows[:5], cols[:5]):
    print(f"  [{r},{c}]: got {product[r,c]:.0f}, expected {expected[r,c]:.0f}")
```

Output:
```
Max error: 4.0

8 positions with large errors:
  [16,7]: got -2, expected 0
  [16,8]: got -1, expected 0
  [16,9]: got -1, expected 0
  [16,10]: got 1, expected 0
  [16,11]: got -2, expected 0
```

## Why This Is A Bug

When `exact=True`, the documentation states the result is "an array of type `numpy.int64` (if n <= 35)" which implies exact integer arithmetic. However, for `n >= 19`, computing `P @ P_inv` silently overflows int64 during intermediate calculations:

- `P` contains values up to ~9 billion (uint64)
- `P_inv` contains values up to ~3 billion (int64)
- Matrix multiplication computes dot products like 9B × 3B = 27×10^18
- This exceeds int64 max (9.2×10^18), causing overflow

Testing with arbitrary precision confirms `invpascal()` computes correct values, but int64 arithmetic corrupts the result:

```python
P_obj = scipy.linalg.pascal(n, exact=True).astype(object)
P_inv_obj = scipy.linalg.invpascal(n, exact=True).astype(object)
product_obj = P_obj @ P_inv_obj
# Result is exact identity with no errors
```

## Fix

The function should use object dtype (arbitrary precision Python integers) for all values of n when `exact=True`, not just n > 34:

```diff
--- a/scipy/linalg/_special_matrices.py
+++ b/scipy/linalg/_special_matrices.py
@@ -500,10 +500,7 @@ def invpascal(n, kind='symmetric', exact=True):

     if kind == 'symmetric':
         if exact:
-            if n > 34:
-                dt = object
-            else:
-                dt = np.int64
+            dt = object
         else:
             dt = np.float64
         invp = np.empty((n, n), dtype=dt)
```

Alternatively, use a more conservative threshold based on when overflow actually occurs:

```diff
         if exact:
-            if n > 34:
+            if n > 18:
                 dt = object
             else:
                 dt = np.int64
```

The second option maintains performance for small matrices while preventing overflow for n >= 19.