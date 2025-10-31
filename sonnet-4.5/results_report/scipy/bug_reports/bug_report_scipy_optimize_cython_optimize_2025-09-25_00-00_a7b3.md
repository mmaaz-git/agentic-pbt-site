# Bug Report: scipy.optimize.cython_optimize Uninitialized Memory in iterations Field

**Target**: `scipy.optimize.cython_optimize._zeros.full_output_example` (and underlying C functions)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `iterations` field in the `zeros_full_output` struct contains uninitialized memory when root-finding functions (brentq, bisect, ridder, brenth) find a root at or very close to a boundary value (xa or xb). This results in garbage values like `-281778880` or `234072704` instead of valid iteration counts.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.optimize.cython_optimize import _zeros

@given(
    a0=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_full_output_has_valid_counts_on_success(a0):
    args = (a0, 0.0, 0.0, 1.0)
    xa, xb = 0.0, 10.0
    xtol, rtol, mitr = 1e-6, 1e-6, 100

    output = _zeros.full_output_example(args, xa, xb, xtol, rtol, mitr)

    if output['error_num'] == 0:
        assert output['iterations'] >= 1, \
            f"Successful solve should have iterations >= 1, got {output['iterations']}"
        assert output['funcalls'] >= 1, \
            f"Successful solve should have funcalls >= 1, got {output['funcalls']}"
        assert output['funcalls'] >= output['iterations'], \
            f"funcalls should be >= iterations"
```

**Failing input**: `a0=0.0` (root at boundary x=0)

## Reproducing the Bug

```python
from scipy.optimize.cython_optimize import _zeros

a0 = 0.0
args = (a0, 0.0, 0.0, 1.0)
xa, xb = 0.0, 10.0
xtol, rtol, mitr = 1e-6, 1e-6, 100

output = _zeros.full_output_example(args, xa, xb, xtol, rtol, mitr)
print(f"iterations={output['iterations']}, funcalls={output['funcalls']}, "
      f"error_num={output['error_num']}, root={output['root']}")

assert output['iterations'] >= 1
```

**Output:**
```
iterations=-281778880, funcalls=2, error_num=0, root=0.0
AssertionError
```

**Additional test cases that trigger the bug:**
```python
test_cases = [
    ("Root at xa=0", (0.0, 0.0, 0.0, 1.0), 0.0, 10.0),
    ("Root at xb=0", (0.0, 0.0, 0.0, 1.0), -5.0, 0.0),
]

for desc, args, xa, xb in test_cases:
    output = _zeros.full_output_example(args, xa, xb, 1e-6, 1e-6, 100)
    print(f"{desc}: iterations={output['iterations']}")
```

## Why This Is A Bug

The `zeros_full_output` struct is documented to contain the number of iterations in its `iterations` field. When a root-finding algorithm successfully finds a root (error_num=0), this field should contain a valid non-negative integer representing the number of iterations performed. Instead, it contains garbage values from uninitialized memory.

This violates the API contract stated in the module documentation:
- `int iterations`: number of iterations

The bug occurs specifically when:
1. The root is found at or very near a boundary (xa or xb)
2. The algorithm terminates early after evaluating the function at the boundaries
3. The `iterations` field is never set/initialized before returning

## Fix

The bug is in the underlying C implementation of the root-finding functions (bisect, ridder, brenth, brentq). When these functions detect that f(xa) or f(xb) is already close enough to zero (within tolerance), they return early without setting the `iterations` field in the `zeros_full_output` struct.

The fix requires modifying the C source code to ensure that all fields of `zeros_full_output` are properly initialized before any early returns. Specifically:

1. Initialize the `zeros_full_output` struct at the beginning of each function
2. OR ensure that the `iterations` field is set in all code paths, including early termination paths

Without access to the C source code in this environment, I cannot provide a specific patch. However, the fix would look conceptually like this:

```c
/* In each root-finding function (bisect, ridder, brenth, brentq) */
double brentq(callback_type f, double xa, double xb, void* args,
              double xtol, double rtol, int iter,
              zeros_full_output *full_output) {

    /* Initialize full_output if provided */
    if (full_output != NULL) {
        full_output->iterations = 0;  /* Add this initialization */
        full_output->funcalls = 0;
        full_output->error_num = 0;
        full_output->root = 0.0;
    }

    /* ... rest of function ... */

    /* In early termination paths, ensure iterations is set: */
    if (fabs(fa) < tolerance) {
        if (full_output != NULL) {
            full_output->root = xa;
            full_output->iterations = 0;  /* Ensure this is set */
            full_output->funcalls = 2;
            full_output->error_num = 0;
        }
        return xa;
    }

    /* ... */
}
```

The critical change is ensuring `full_output->iterations` is set to a valid value (likely 0 or 1) in all code paths, especially when the root is found at a boundary.