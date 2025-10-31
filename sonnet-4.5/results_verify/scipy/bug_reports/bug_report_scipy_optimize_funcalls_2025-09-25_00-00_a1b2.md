# Bug Report: scipy.optimize Bracketing Methods Under-Report Function Calls

**Target**: `scipy.optimize.{bisect, ridder, brenth, brentq}`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The bracketing root-finding methods (`bisect`, `ridder`, `brenth`, `brentq`) under-report the actual number of function evaluations by 2. The methods make initial calls to `f(a)` and `f(b)` to verify the sign condition, but these calls are not included in the `function_calls` field of the returned `RootResults` object.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from scipy.optimize import bisect


@given(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=500)
def test_bisect_funcalls_accurate(a, b):
    assume(abs(a - b) > 1e-8)
    assume(a < b)

    call_count = [0]

    def f(x):
        call_count[0] += 1
        return x - 1.5

    fa, fb = f(a), f(b)
    assume(fa * fb < 0)

    call_count[0] = 0
    root, result = bisect(f, a, b, full_output=True, disp=False)

    actual_calls = call_count[0]
    reported_calls = result.function_calls

    assert actual_calls == reported_calls, \
        f"bisect: reported {reported_calls} calls but actually made {actual_calls}"
```

**Failing input**: Any valid interval where `f(a)` and `f(b)` have opposite signs

## Reproducing the Bug

```python
from scipy.optimize import bisect

call_count = 0

def f(x):
    global call_count
    call_count += 1
    print(f"Call {call_count}: f({x})")
    return x - 1.5

a, b = 0.0, 2.0
root, result = bisect(f, a, b, full_output=True, disp=False)

print(f"Actual function calls: {call_count}")
print(f"Reported function_calls: {result.function_calls}")
```

Output:
```
Call 1: f(0.0)
Call 2: f(2.0)
Call 3: f(1.0)
Call 4: f(1.5)
Call 5: f(1.5)
Call 6: f(1.5)
Actual function calls: 6
Reported function_calls: 4
```

## Why This Is A Bug

The `RootResults.function_calls` field is documented to contain "Number of times the function was called." This should include ALL function evaluations, including the initial checks of `f(a)` and `f(b)` that verify the bracketing condition `f(a) * f(b) < 0`.

Users relying on `function_calls` for performance analysis or billing (e.g., expensive objective functions) will underestimate the actual computational cost.

The issue affects `bisect`, `ridder`, `brenth`, and `brentq`.

## Fix

The C implementation in `scipy/optimize/_zeros.c` (or the Cython wrapper) needs to be updated to include the initial function evaluations in the returned `funcalls` count. The fix should ensure that the initial evaluations of `f(a)` and `f(b)` are counted.

A potential fix location is in the C function that implements each solver, where it should increment the function call counter before or immediately after the initial evaluations.