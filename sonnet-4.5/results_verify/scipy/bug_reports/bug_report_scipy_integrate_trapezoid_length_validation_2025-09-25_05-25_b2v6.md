# Bug Report: scipy.integrate.trapezoid Length Validation

**Target**: `scipy.integrate.trapezoid`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.integrate.trapezoid` does not validate that `x` and `y` arrays have matching lengths, leading to inconsistent behavior: sometimes silently producing incorrect results, sometimes raising obscure error messages. This contrasts with `scipy.integrate.simpson` and `scipy.integrate.cumulative_trapezoid`, which properly validate input lengths.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from scipy import integrate

@given(
    y=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=2, max_size=20),
    x=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=2, max_size=20)
)
@settings(max_examples=300)
def test_trapezoid_handles_mismatched_lengths(y, x):
    if len(y) == len(x):
        return

    y_arr = np.array(y)
    x_arr = np.array(x)

    try:
        result = integrate.trapezoid(y_arr, x=x_arr)
        assert False, f"Should have raised error for mismatched lengths: len(y)={len(y)}, len(x)={len(x)}"
    except (ValueError, IndexError) as e:
        pass
```

**Failing input**: `y=[0.0, 0.0], x=[0.0, 0.0, 0.0]`

## Reproducing the Bug

```python
import numpy as np
from scipy import integrate

print("Case 1: y shorter than x - silently produces wrong result")
y = np.array([1.0, 2.0])
x = np.array([0.0, 1.0, 2.0, 3.0])
result = integrate.trapezoid(y, x=x)
print(f"y = {y} (length {len(y)})")
print(f"x = {x} (length {len(x)})")
print(f"Result: {result}")

print("\nCase 2: y longer than x - produces obscure broadcast error")
y = np.array([1.0, 2.0, 3.0, 4.0])
x = np.array([0.0, 1.0, 2.0])
try:
    result = integrate.trapezoid(y, x=x)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nCompare with simpson - it validates correctly:")
y = np.array([1.0, 2.0, 3.0])
x = np.array([0.0, 1.0])
try:
    result = integrate.simpson(y, x=x)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nCompare with cumulative_trapezoid - it also validates correctly:")
y = np.array([1.0, 2.0])
x = np.array([0.0, 1.0, 2.0, 3.0])
try:
    result = integrate.cumulative_trapezoid(y, x=x)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

Output:
```
Case 1: y shorter than x - silently produces wrong result
y = [1. 2.] (length 2)
x = [0. 1. 2. 3.] (length 4)
Result: 4.5

Case 2: y longer than x - produces obscure broadcast error
y = [1. 2. 3. 4.] (length 4)
x = [0. 1. 2.] (length 3)
Error: ValueError: operands could not be broadcast together with shapes (2,) (3,)

Compare with simpson - it validates correctly:
y = [1. 2. 3.] (length 3)
x = [0. 1.] (length 2)
Error: ValueError: If given, length of x along axis must be the same as y.

Compare with cumulative_trapezoid - it also validates correctly:
y = [1. 2.] (length 2)
x = [0. 1. 2. 3.] (length 4)
Error: ValueError: If given, length of x along axis must be the same as y.
```

## Why This Is A Bug

1. **Silent incorrect results**: When `len(y) < len(x)`, the function produces a result without error, but the mathematical interpretation is unclear - it's not a valid trapezoidal integration.

2. **Inconsistent behavior**: Sometimes accepts mismatched inputs, sometimes rejects them.

3. **Poor error messages**: When it does fail, the error message mentions "broadcast" rather than the actual problem (length mismatch).

4. **API inconsistency**: Both `simpson` and `cumulative_trapezoid` correctly validate input lengths with clear error messages, but `trapezoid` does not.

## Fix

Add input validation at the beginning of the `trapezoid` function, similar to what `simpson` and `cumulative_trapezoid` already do:

```diff
def trapezoid(y, x=None, dx=1.0, axis=-1):
    y = np.asanyarray(y)

    if x is not None:
        x = np.asanyarray(x)
+       if x.shape[axis] != y.shape[axis]:
+           raise ValueError(
+               f"If given, length of x ({x.shape[axis]}) along axis "
+               f"must be the same as y ({y.shape[axis]})."
+           )
        d = np.diff(x, axis=axis)
    else:
        d = dx
```