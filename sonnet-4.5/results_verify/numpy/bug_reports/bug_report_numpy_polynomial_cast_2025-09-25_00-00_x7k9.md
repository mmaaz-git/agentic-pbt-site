# Bug Report: numpy.polynomial cast() Method Crashes When Used as Instance Method

**Target**: `numpy.polynomial.Polynomial.cast()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cast()` classmethod crashes with a confusing AttributeError when called as an instance method (e.g., `polynomial_instance.cast(TargetClass)`), even though Python allows classmethods to be called on instances.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from numpy.polynomial import Polynomial

polynomial_coefs = st.lists(
    st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
    min_size=1,
    max_size=6
)

@given(polynomial_coefs)
@settings(max_examples=300)
def test_cast_to_same_type(coefs):
    p = Polynomial(coefs)
    p_cast = p.cast(Polynomial)

    assert np.allclose(p.coef, p_cast.coef, rtol=1e-10, atol=1e-10)
```

**Failing input**: `coefs=[0.0]` (or any coefficient list)

## Reproducing the Bug

```python
from numpy.polynomial import Polynomial

p = Polynomial([1, 2, 3])
p_cast = p.cast(Polynomial)
```

Output:
```
AttributeError: 'numpy.ndarray' object has no attribute 'window'
```

## Why This Is A Bug

The `cast()` method is a classmethod, which Python allows to be called on instances. When called as `instance.cast(TargetClass)`, it misinterprets `TargetClass` as the series to convert, then attempts to call `convert()` on the class rather than an instance, causing a confusing crash.

**Root cause**: In `_polybase.py` line 1191:
```python
return series.convert(domain, cls, window)
```

When `cast()` is called as instance method with `p.cast(Polynomial)`:
- `series` becomes `Polynomial` (the class, not instance)
- Calls `Polynomial.convert(array([-1,1]), Polynomial, array([-1,1]))`
- Since `Polynomial` is a class, `convert()` binds `self=array([-1,1])` (first positional arg)
- Arguments shift: `domain=Polynomial`, `kind=array([-1,1])`, `window=None`
- When `convert()` executes `window = kind.window`, `kind` is the array, causing AttributeError

## Fix

**Option 1**: Use keyword arguments in cast() to prevent argument shifting:

```diff
--- a/numpy/polynomial/_polybase.py
+++ b/numpy/polynomial/_polybase.py
@@ -1188,7 +1188,7 @@ class ABCPolyBase(abc.ABC):
             domain = cls.domain
         if window is None:
             window = cls.window
-        return series.convert(domain, cls, window)
+        return series.convert(domain=domain, kind=cls, window=window)
```

**Option 2**: Add validation to ensure `series` is an instance:

```diff
--- a/numpy/polynomial/_polybase.py
+++ b/numpy/polynomial/_polybase.py
@@ -1184,6 +1184,8 @@ class ABCPolyBase(abc.ABC):
         convert : similar instance method

         """
+        if not hasattr(series, 'convert'):
+            raise TypeError(f"series must be a polynomial instance, got {type(series)}")
         if domain is None:
             domain = cls.domain
         if window is None:
```

**Recommendation**: Option 1 is simpler and more robust.