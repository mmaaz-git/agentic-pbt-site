# Bug Report: scipy.optimize.cython_optimize._zeros.full_output_example Returns Uninitialized Memory

**Target**: `scipy.optimize.cython_optimize._zeros.full_output_example`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `full_output_example` function in `scipy.optimize.cython_optimize._zeros` returns uninitialized memory and invalid error codes when given an incorrect number of arguments, instead of raising a proper exception.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pytest


@given(
    arg_length=st.integers(min_value=0, max_value=10),
    xa=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    xb=st.floats(min_value=1.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_full_output_example_validates_args_length(arg_length, xa, xb):
    from scipy.optimize.cython_optimize import _zeros

    args = tuple(1.0 for _ in range(arg_length))
    xtol, rtol, mitr = 1e-6, 1e-6, 100

    try:
        result = _zeros.full_output_example(args, xa, xb, xtol, rtol, mitr)

        if isinstance(result, dict):
            error_num = result.get('error_num')

            if error_num not in [-2, -1, 0]:
                pytest.fail(f"error_num should be -2, -1, or 0, got {error_num}")

    except (TypeError, IndexError, ValueError) as e:
        pass
```

**Failing input**: `arg_length=0, xa=0.0, xb=1.0` (or any arg_length != 4)

## Reproducing the Bug

```python
from scipy.optimize.cython_optimize import _zeros

args = ()
xa, xb = 0.0, 1.0
xtol, rtol, mitr = 1e-6, 1e-6, 100

result = _zeros.full_output_example(args, xa, xb, xtol, rtol, mitr)
print(f"Result: {result}")
```

Output:
```
Result: {'funcalls': 0, 'iterations': -1755511856, 'error_num': 30443, 'root': 0.0}
```

The `error_num` value of 30443 is not one of the documented values (-2, -1, or 0), and the `iterations` value of -1755511856 is clearly uninitialized memory.

## Why This Is A Bug

1. **Contract violation**: According to the scipy documentation, `error_num` should be -1 (sign error), -2 (convergence error), or 0 (success). Instead, the function returns garbage values like 30443.

2. **Uninitialized memory exposure**: The function returns uninitialized memory values for `iterations` (e.g., -1755511856), which is a security and correctness issue.

3. **Suppressed exceptions**: The function raises `IndexError` in an unraisable context (inside Cython cleanup code), which means the error is silently suppressed instead of being propagated to the caller.

4. **Missing input validation**: The function expects exactly 4 arguments but doesn't validate this requirement, leading to undefined behavior.

## Fix

The bug occurs because the Cython code tries to convert a Python tuple to a C array of size 4 without proper validation. When the tuple has the wrong size, the conversion fails in a cleanup context where exceptions cannot be raised.

The fix should add explicit input validation before attempting the conversion:

```python
def full_output_example(args, xa, xb, xtol, rtol, mitr):
    if not isinstance(args, tuple):
        raise TypeError(f"args must be a tuple, got {type(args).__name__}")

    if len(args) != 4:
        raise ValueError(f"args must have exactly 4 elements, got {len(args)}")

    # ... rest of the function
```

Alternatively, if the function is meant to be flexible about the number of arguments, the Cython code should be updated to handle variable-length inputs properly without accessing uninitialized memory.