# Bug Report: scipy.optimize.cython_optimize Iteration Count Corruption

**Target**: `scipy.optimize.cython_optimize._zeros.full_output_example`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `full_output_example` function returns corrupted iteration counts in the output dictionary. The `iterations` field contains nonsensical values including negative numbers and values that vastly exceed the maximum iteration limit. The values are also non-deterministic, changing between runs with identical inputs.

## Property-Based Test

```python
from hypothesis import assume, given, settings, strategies as st
from scipy.optimize.cython_optimize import _zeros


@given(
    st.tuples(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    ),
    st.floats(min_value=-1000, max_value=1000),
    st.floats(min_value=-1000, max_value=1000),
    st.integers(min_value=1, max_value=10000),
)
@settings(max_examples=1000)
def test_iteration_count_validity(coeffs, xa, xb, mitr):
    assume(xa < xb)
    xtol, rtol = 1e-6, 1e-6

    result = _zeros.full_output_example(coeffs, xa, xb, xtol, rtol, mitr)
    iterations = result['iterations']
    funcalls = result['funcalls']

    assert iterations >= 0, f'Iteration count {iterations} is negative'
    assert iterations <= mitr, f'Iteration count {iterations} exceeds max iterations {mitr}'
    assert funcalls >= 0, f'Function call count {funcalls} is negative'
```

**Failing input**: Multiple inputs fail, including `coeffs=(0.0, 0.0, 0.0, 0.0), xa=0.01, xb=1.0, mitr=100`

## Reproducing the Bug

```python
from scipy.optimize.cython_optimize import _zeros

coeffs = (0.0, 0.0, 0.0, 0.0)
xa, xb = 0.01, 1.0
xtol, rtol = 1e-6, 1e-6
mitr = 100

for i in range(10):
    result = _zeros.full_output_example(coeffs, xa, xb, xtol, rtol, mitr)
    print(f'Run {i+1}: iterations={result["iterations"]}, funcalls={result["funcalls"]}')
```

Output:
```
Run 1: iterations=-1160101440, funcalls=2
Run 2: iterations=1, funcalls=2
Run 3: iterations=9982, funcalls=2
Run 4: iterations=1, funcalls=2
Run 5: iterations=9982, funcalls=2
Run 6: iterations=9982, funcalls=2
Run 7: iterations=9982, funcalls=2
Run 8: iterations=9982, funcalls=2
Run 9: iterations=9982, funcalls=2
Run 10: iterations=9982, funcalls=2
```

## Why This Is A Bug

The `iterations` field in the full output should contain the actual number of iterations performed by the root-finding algorithm. This value should always be:
1. Non-negative (iterations cannot be negative)
2. Less than or equal to `mitr` (the maximum iteration limit)
3. Deterministic (same inputs should produce same outputs)

The observed behavior violates all three invariants:
- Negative values (e.g., -1160101440) indicate memory corruption or uninitialized variables
- Values exceeding mitr (e.g., 9982 when mitr=100) indicate incorrect bookkeeping
- Non-deterministic behavior (different values across runs with identical inputs) suggests uninitialized memory is being read

This corruption likely stems from uninitialized memory in the Cython code that constructs the full output structure. The `funcalls` field appears to be correct (always 2 in this case), which suggests the issue is specific to the `iterations` field initialization or assignment.

## Fix

The bug is likely in the Cython source code for `_zeros.pyx` where the `zeros_full_output` struct is populated. The `iterations` field is probably not being properly initialized or copied from the internal solver state. Without access to the source code, I cannot provide a specific patch, but the fix would involve:

1. Ensuring the `zeros_full_output.iterations` field is properly initialized before being returned
2. Ensuring the iteration counter in the root-finding algorithm is correctly copied to the output struct
3. Adding validation/sanitization to ensure the iteration count is within valid bounds before returning