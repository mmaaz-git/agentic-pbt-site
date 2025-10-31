# Bug Report: numpy.polynomial.Polynomial.cast() Crashes When Called as Instance Method

**Target**: `numpy.polynomial.Polynomial.cast()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cast()` classmethod crashes with AttributeError when called as an instance method (e.g., `polynomial_instance.cast(TargetClass)`), even though Python allows classmethods to be called on instances.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from numpy.polynomial import Polynomial
import numpy as np

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

if __name__ == "__main__":
    test_cast_to_same_type()
```

<details>

<summary>
**Failing input**: `coefs=[0.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 20, in <module>
    test_cast_to_same_type()
    ~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 12, in test_cast_to_same_type
    @settings(max_examples=300)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 15, in test_cast_to_same_type
    p_cast = p.cast(Polynomial)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/polynomial/_polybase.py", line 1191, in cast
    return series.convert(domain, cls, window)
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/polynomial/_polybase.py", line 813, in convert
    window = kind.window
             ^^^^^^^^^^^
AttributeError: 'numpy.ndarray' object has no attribute 'window'
Falsifying example: test_cast_to_same_type(
    coefs=[0.0],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from numpy.polynomial import Polynomial

# Create a simple polynomial
p = Polynomial([1, 2, 3])

# Try to use cast() as an instance method
# (This should work since classmethods can be called on instances in Python)
print("Calling p.cast(Polynomial)...")
p_cast = p.cast(Polynomial)
print("Success:", p_cast)
```

<details>

<summary>
AttributeError: 'numpy.ndarray' object has no attribute 'window'
</summary>
```
Calling p.cast(Polynomial)...
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/29/repo.py", line 9, in <module>
    p_cast = p.cast(Polynomial)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/polynomial/_polybase.py", line 1191, in cast
    return series.convert(domain, cls, window)
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/polynomial/_polybase.py", line 813, in convert
    window = kind.window
             ^^^^^^^^^^^
AttributeError: 'numpy.ndarray' object has no attribute 'window'
```
</details>

## Why This Is A Bug

Python allows classmethods to be called on instances. When a classmethod is called via an instance (e.g., `instance.classmethod(args)`), Python automatically passes the class (not the instance) as the first argument. This is standard Python behavior since Python 2.2.

However, when `cast()` is called as `p.cast(Polynomial)`, the implementation incorrectly handles the arguments:

1. Python binds `p.__class__` (i.e., `Polynomial`) as `cls` (first argument)
2. The user's `Polynomial` argument becomes `series` (second positional argument)
3. Inside `cast()` at line 1191, it executes: `return series.convert(domain, cls, window)`
4. Since `series` is now the `Polynomial` class (not an instance), `convert()` gets called on the class
5. When `convert()` is called on a class, Python binds the first positional argument (`domain`) as `self`
6. This shifts all arguments: `self=domain` (an ndarray), `domain=cls`, `kind=window`
7. At line 813 in `convert()`, it tries to access `kind.window`, but `kind` is now the ndarray from the shifted arguments
8. This causes the AttributeError: 'numpy.ndarray' object has no attribute 'window'

The documentation for `cast()` states it "Convert series to series of this class" and mentions `convert()` as the "similar instance method". Nothing in the documentation prohibits calling `cast()` on an instance, and Python's classmethod semantics support this usage.

## Relevant Context

- **Source location**: `/numpy/polynomial/_polybase.py`, lines 1153-1191 (cast method), lines 779-814 (convert method)
- **Documentation**: The `cast()` method is documented as a `@classmethod` that converts series between polynomial types
- **Related method**: `convert()` is the instance method that performs the actual conversion
- **Python behavior**: Classmethods can be called on instances - see [Python documentation on classmethod](https://docs.python.org/3/library/functions.html#classmethod)
- **Workaround**: Users can call `Polynomial.cast(p, Polynomial)` instead of `p.cast(Polynomial)`

## Proposed Fix

Use keyword arguments when calling `convert()` to prevent argument shifting when `cast()` is called on an instance:

```diff
--- a/numpy/polynomial/_polybase.py
+++ b/numpy/polynomial/_polybase.py
@@ -1188,5 +1188,5 @@ class ABCPolyBase(abc.ABC):
             domain = cls.domain
         if window is None:
             window = cls.window
-        return series.convert(domain, cls, window)
+        return series.convert(domain=domain, kind=cls, window=window)
```