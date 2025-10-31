# Bug Report: numpy.polynomial.polymul Associativity Violation

**Target**: `numpy.polynomial.polynomial.polymul`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`polymul` violates associativity when multiplying polynomials with denormal (subnormal) coefficients. Due to inconsistent handling of floating-point underflow and trailing zero trimming, `(c1 * c2) * c3` can produce a different result than `c1 * (c2 * c3)`, breaking a fundamental mathematical property.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from numpy.polynomial import polynomial as P

coefficients = st.lists(
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    min_size=1,
    max_size=10
).map(lambda x: np.array(x))

@settings(max_examples=500)
@given(c1=coefficients, c2=coefficients, c3=coefficients)
def test_polymul_associative(c1, c2, c3):
    result1 = P.polymul(P.polymul(c1, c2), c3)
    result2 = P.polymul(c1, P.polymul(c2, c3))
    np.testing.assert_allclose(result1, result2, rtol=1e-9, atol=1e-9)
```

**Failing input**:
- `c1 = np.array([2.0])`
- `c2 = np.array([0.0, 5e-324])`
- `c3 = np.array([0.5])`

## Reproducing the Bug

```python
import numpy as np
from numpy.polynomial import polynomial

c1 = np.array([2.0])
c2 = np.array([0.0, 5e-324])
c3 = np.array([0.5])

result_ltr = polynomial.polymul(polynomial.polymul(c1, c2), c3)
result_rtl = polynomial.polymul(c1, polynomial.polymul(c2, c3))

print("(c1 * c2) * c3 =", result_ltr)
print("c1 * (c2 * c3) =", result_rtl)
print("Equal?", np.array_equal(result_ltr, result_rtl))
```

Output:
```
(c1 * c2) * c3 = [0.e+000 5.e-324]
c1 * (c2 * c3) = [0.]
Equal? False
```

## Why This Is A Bug

Polynomial multiplication must be associative: `(p * q) * r = p * (q * r)` for all polynomials p, q, r. This is a fundamental algebraic property that users rely on.

The bug occurs because:
1. `c1 * c2 = [0.0, 1e-323]` preserves the denormal number
2. `[0.0, 1e-323] * c3 = [0.0, 5e-324]` produces a denormal result that is kept
3. But `c2 * c3 = [0.0, 2.5e-324]` where `2.5e-324` underflows to exactly `0.0`, triggering `polytrim` to remove it, yielding `[0.]`
4. Thus `c1 * [0.] = [0.]`

The root cause is that `polymul` calls `polytrim` after multiplication, and `polytrim` removes trailing zeros. When a subnormal coefficient underflows to exactly 0.0 during multiplication, it gets trimmed. But when the same mathematical value is reached via a different sequence of operations that preserves denormal numbers, it doesn't get trimmed.

## Fix

The fix would require consistent handling of denormal numbers in `polymul`. One approach:

1. Normalize all denormal coefficients below a threshold (e.g., `np.finfo(float).tiny`) to exactly 0.0 before trimming
2. Or document that associativity is not guaranteed for coefficients near machine epsilon

This would require examining the `polymul` implementation and ensuring consistent trimming behavior.