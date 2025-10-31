# Bug Report: scipy.odr.polynomial Pickle Failure

**Target**: `scipy.odr.polynomial`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Models created with `scipy.odr.polynomial()` cannot be pickled, while all other predefined ODR models (multilinear, exponential, quadratic, unilinear) can be pickled successfully. This is inconsistent and breaks the expected behavior that Model objects should be serializable.

## Property-Based Test

```python
import scipy.odr
import pickle
from hypothesis import given, settings
from hypothesis import strategies as st


@given(
    n_points=st.integers(min_value=10, max_value=50),
    seed=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=50)
def test_pickling_round_trip(n_points, seed):
    poly_model = scipy.odr.polynomial(2)
    model_pickled = pickle.loads(pickle.dumps(poly_model))
    assert model_pickled is not None
```

**Failing input**: Any polynomial model created with `scipy.odr.polynomial(n)` for any n

## Reproducing the Bug

```python
import scipy.odr
import pickle

poly_model = scipy.odr.polynomial(2)
pickle.dumps(poly_model)
```

Output:
```
AttributeError: Can't get local object 'polynomial.<locals>._poly_est'
```

All other predefined models work fine:
```python
import scipy.odr
import pickle

pickle.dumps(scipy.odr.multilinear)
pickle.dumps(scipy.odr.exponential)
pickle.dumps(scipy.odr.quadratic)
pickle.dumps(scipy.odr.unilinear)
```

## Why This Is A Bug

1. **Inconsistency**: All other predefined ODR models (multilinear, exponential, quadratic, unilinear) can be pickled successfully
2. **Documentation**: The scipy.odr documentation and existing tests show that Model objects are expected to be pickleable
3. **User Impact**: Users cannot save and reload their polynomial models, which is a common workflow in scientific computing
4. **Breaking Expectation**: There's no indication in the documentation that polynomial models would behave differently from other models

## Fix

The issue is that `polynomial()` returns a Model with a local function `_poly_est` as the estimate function. Local functions cannot be pickled. The fix is to move `_poly_est` to module level or make it a method of a class.

Looking at the implementation in `_models.py`, the `polynomial` function should be refactored to avoid using local functions. Here's the general approach:

```diff
--- a/scipy/odr/_models.py
+++ b/scipy/odr/_models.py
@@ -100,12 +100,15 @@ def _poly_fcn(B, x, powers):
     return np.sum(B[:, np.newaxis] * (x[np.newaxis, :] ** powers[:, np.newaxis]), axis=0)


+def _poly_est(data, powers):
+    # Move this outside of polynomial() function
+    # Implementation moved from local function to module level
+    ...
+
 def polynomial(order):
     ...
-    def _poly_est(data):
-        ...
-        return est
-    return Model(fcn, fjacb=fjd, fjacd=fjd_x, estimate=_poly_est)
+    # Create a partial function or wrapper that captures powers
+    estimate_fn = lambda data: _poly_est(data, powers)
+    return Model(fcn, fjacb=fjd, fjacd=fjd_x, estimate=estimate_fn)
```

Alternatively, use `functools.partial` or create a pickleable wrapper class for the estimate function.