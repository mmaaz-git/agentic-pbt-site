# Bug Report: scipy.optimize.cython_optimize full_output_example Returns Corrupted Data

**Target**: `scipy.optimize.cython_optimize._zeros.full_output_example`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `full_output_example` function returns corrupted data in its output dictionary, including nonsensical iteration counts and negative error numbers, even with valid inputs.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import scipy.optimize.cython_optimize._zeros as zeros


def polynomial(x, c0, c1, c2, c3):
    return c0 + c1*x + c2*x^2 + c3*x^3


@given(
    c0=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    c1=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    c2=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    c3=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False).filter(lambda x: abs(x) > 0.01),
)
def test_full_output_validity(c0, c1, c2, c3):
    xa, xb = -10.0, 10.0
    mitr = 100

    f_xa = polynomial(xa, c0, c1, c2, c3)
    f_xb = polynomial(xb, c0, c1, c2, c3)
    assume(f_xa * f_xb < 0)

    result = zeros.full_output_example((c0, c1, c2, c3), xa, xb, 1e-6, 1e-6, mitr)

    assert result['iterations'] >= 0, f"iterations should be non-negative, got {result['iterations']}"
    assert result['iterations'] <= mitr, f"iterations {result['iterations']} exceeds max {mitr}"
    assert result['error_num'] >= 0, f"error_num should be non-negative, got {result['error_num']}"
```

**Failing input**: Most inputs with a root in the interval

## Reproducing the Bug

```python
import scipy.optimize.cython_optimize._zeros as zeros

args = (-2.0, 0.0, 0.0, 1.0)
result = zeros.full_output_example(args, 1.0, 2.0, 1e-9, 1e-9, 100)

print(f"Result: {result}")
print(f"Iterations: {result['iterations']}")
print(f"Max iterations: 100")
print(f"Error number: {result['error_num']}")
```

Output:
```
Result: {'funcalls': 2, 'iterations': 1027459584, 'error_num': -1, 'root': 1.2599210498948732}
Iterations: 1027459584
Max iterations: 100
Error number: -1
```

## Why This Is A Bug

1. **Corrupted iteration count**: Returns 1027459584 iterations when max_iterations is 100. This appears to be uninitialized memory or a corrupted value.
2. **Negative error number**: Returns -1 for error_num, which typically indicates an error state, despite finding the correct root.
3. **Inconsistent with root quality**: The root itself is correct (1.259...), but the metadata suggests failure.
4. **Values change between runs**: Different runs with same inputs produce different garbage values for iterations (e.g., -45849920, 1027459584).

## Fix

The full_output_example function appears to have memory corruption or uninitialized variable issues. The wrapper code needs to properly initialize all output struct members before calling the underlying C function, or the C function needs to properly set all output fields.

Likely issue in the Cython wrapper:
- Output struct members not initialized
- Incorrect struct field ordering between Python and C
- Memory not properly allocated for the output dictionary