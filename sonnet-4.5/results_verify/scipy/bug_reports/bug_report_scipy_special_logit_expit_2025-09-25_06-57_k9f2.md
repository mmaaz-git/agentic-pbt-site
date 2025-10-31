# Bug Report: scipy.special logit/expit Composition Returns Incorrect Values

**Target**: `scipy.special.logit` and `scipy.special.expit`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The composition `logit(expit(x))` returns incorrect values for x >= 27 and completely fails (returns inf) for x >= 40, despite documentation claiming these functions are inverses.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import scipy.special as sp
import math

@given(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
def test_logit_expit_inverse(x):
    result = sp.logit(sp.expit(x))
    assert math.isclose(result, x, rel_tol=1e-6), \
        f"logit(expit({x})) = {result}, expected {x}"
```

**Failing inputs**: x=27.0, x=30.0, x=35.0, x=40.0, and all x >= 40

## Reproducing the Bug

```python
import numpy as np
from scipy.special import expit, logit

test_values = [5.0, 20.0, 27.0, 30.0, 35.0, 40.0]

for x in test_values:
    result = logit(expit(x))
    error = abs(result - x) if np.isfinite(result) else float('inf')
    print(f"x={x:4.1f}: logit(expit(x))={result:12.6f}, error={error:.2e}")
```

Output:
```
x= 5.0: logit(expit(x))=    5.000000, error=1.87e-14
x=20.0: logit(expit(x))=   20.000000, error=3.59e-08
x=27.0: logit(expit(x))=   26.999958, error=4.19e-05
x=30.0: logit(expit(x))=   30.001021, error=1.02e-03
x=35.0: logit(expit(x))=   34.945041, error=5.50e-02
x=40.0: logit(expit(x))=         inf, error=inf
```

## Why This Is A Bug

The `scipy.special.expit` documentation explicitly states: "It is the inverse of the logit function."

The documentation also provides an example demonstrating the inverse relationship:
```python
>>> logit(expit([-2.5, 0, 3.1, 5.0]))
array([-2.5,  0. ,  3.1,  5. ])
```

However, this inverse property breaks down for moderate values of x:
- For x >= 27: significant numerical error (>1e-5)
- For x >= 35: large error (>5%)
- For x >= 40: complete failure (returns inf instead of x)

This violates the documented behavior and would mislead users who expect these functions to be true inverses.

## Root Cause

For large positive x, `expit(x) = 1/(1+exp(-x))` becomes extremely close to 1.0. When x >= 40, it rounds to exactly 1.0 in floating point arithmetic. Subsequently, `logit(1.0) = log(1/(1-1))` correctly returns inf, but this is not the mathematically correct result for `logit(expit(40))`.

The issue is that the intermediate value `expit(x)` loses precision for large x, and this precision loss is not recoverable in the logit computation.

## Fix

The issue could be addressed in several ways:

**Option 1**: Improve numerical stability of `logit` for values very close to 1.0 by using log1p or similar techniques

**Option 2**: Document the numerical limitations in the docstring, e.g.:

```diff
 The expit function, also known as the logistic sigmoid function, is
 defined as ``expit(x) = 1/(1+exp(-x))``.  It is the inverse of the
 logit function.
+
+.. note::
+   Due to floating point precision limitations, the composition
+   ``logit(expit(x))`` may have significant numerical error for |x| > 25
+   and will return inf for x >= 40. For better numerical stability,
+   avoid composing these functions when x is large.
```

**Option 3**: Implement special handling in logit to detect and handle values extremely close to 0 or 1, possibly by maintaining higher precision internally or using different numerical techniques.