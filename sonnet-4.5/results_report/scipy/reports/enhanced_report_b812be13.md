# Bug Report: scipy.optimize.cython_optimize._zeros.loop_example Returns Invalid Root Without Error on Same-Sign Boundaries

**Target**: `scipy.optimize.cython_optimize._zeros.loop_example`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `loop_example` function silently returns an invalid root when the function values at the boundaries have the same sign, violating the fundamental requirement that f(xa) * f(xb) < 0 for bracketing root-finding methods.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings, example
from scipy.optimize.cython_optimize import _zeros
import scipy.optimize as opt
import math


@settings(max_examples=500)
@example(method='bisect', a0=1.0, a3=1.0, xa=0.0, xb=2.0)  # Known failing case
@given(
    method=st.sampled_from(['bisect', 'ridder', 'brenth', 'brentq']),
    a0=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    a3=st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False),
    xa=st.floats(min_value=-100, max_value=0, allow_nan=False, allow_infinity=False),
    xb=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
)
def test_cython_vs_python_same_sign_behavior(method, a0, a3, xa, xb):
    """Test that Cython and Python versions handle same-sign inputs consistently."""
    assume(xa < xb)
    assume(abs(xb - xa) > 0.1)

    xtol, rtol, mitr = 1e-6, 1e-6, 100

    def f(x):
        return a3 * x**3 + a0

    f_xa = f(xa)
    f_xb = f(xb)

    assume(not math.isnan(f_xa) and not math.isnan(f_xb))
    assume(not math.isinf(f_xa) and not math.isinf(f_xb))

    # Test case where signs are the same (should raise error)
    if f_xa * f_xb > 0:
        # Python version should raise ValueError
        python_func = getattr(opt, method)
        python_raised_error = False
        try:
            python_result = python_func(f, xa, xb, xtol=xtol, rtol=rtol, maxiter=mitr, full_output=False, disp=False)
        except ValueError:
            python_raised_error = True

        # Cython version should also raise an error or return an error status
        cython_raised_error = False
        cython_result = None
        try:
            cython_result = _zeros.loop_example(method, (a0,), (0.0, 0.0, a3), xa, xb, xtol, rtol, mitr)
            cython_root = list(cython_result)[0]
            # Check if the returned value is actually a root
            if abs(f(cython_root)) > max(xtol, rtol * abs(cython_root)):
                # Not a valid root - this is the bug!
                cython_raised_error = False  # No error was raised but should have been
        except ValueError:
            cython_raised_error = True

        # Both should handle same-sign inputs the same way
        assert python_raised_error == True, "Python should raise ValueError for same-sign inputs"
        assert cython_raised_error == True or cython_result is None, f"Cython should raise error for same-sign inputs, but returned {cython_result}"

# Run the test
if __name__ == "__main__":
    test_cython_vs_python_same_sign_behavior()
```

<details>

<summary>
**Failing input**: `method='bisect', a0=1.0, a3=1.0, xa=0.0, xb=2.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 61, in <module>
    test_cython_vs_python_same_sign_behavior()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 8, in test_cython_vs_python_same_sign_behavior
    @example(method='bisect', a0=1.0, a3=1.0, xa=0.0, xb=2.0)  # Known failing case
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 57, in test_cython_vs_python_same_sign_behavior
    assert cython_raised_error == True or cython_result is None, f"Cython should raise error for same-sign inputs, but returned {cython_result}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Cython should raise error for same-sign inputs, but returned <map object at 0x760bdcefba90>
Falsifying explicit example: test_cython_vs_python_same_sign_behavior(
    method='bisect',
    a0=1.0,
    a3=1.0,
    xa=0.0,
    xb=2.0,
)
```
</details>

## Reproducing the Bug

```python
from scipy.optimize.cython_optimize import _zeros
import scipy.optimize as opt

xa, xb = 0.0, 2.0
xtol, rtol, mitr = 1e-6, 1e-6, 100

def f(x):
    return x**3 + 1.0

print(f"f({xa}) = {f(xa)}")
print(f"f({xb}) = {f(xb)}")
print(f"Note: Both values are positive (same sign)")

print("\nPython bisect:")
try:
    result = opt.bisect(f, xa, xb, xtol=xtol, rtol=rtol, maxiter=mitr)
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  Correctly raised ValueError: {e}")

print("\nCython loop_example:")
cython_result = _zeros.loop_example('bisect', (1.0,), (0.0, 0.0, 1.0), xa, xb, xtol, rtol, mitr)
root = list(cython_result)[0]
print(f"  Result: {root}")
print(f"  f(root) = {f(root)}")
if abs(f(root)) > xtol:
    print(f"  BUG: Returned non-root without error!")
else:
    print(f"  Valid root found")

print("\nCython full_output_example:")
output = _zeros.full_output_example((1.0, 0.0, 0.0, 1.0), xa, xb, xtol, rtol, mitr)
print(f"  error_num: {output['error_num']}")
print(f"  root: {output['root']}")
print(f"  iterations: {output['iterations']}")
if 'function_calls' in output:
    print(f"  function_calls: {output['function_calls']}")
if output['error_num'] == -1:
    print(f"  Note: full_output correctly sets error_num=-1 (sign error)")
```

<details>

<summary>
Output showing the bug
</summary>
```
f(0.0) = 1.0
f(2.0) = 9.0
Note: Both values are positive (same sign)

Python bisect:
  Correctly raised ValueError: f(a) and f(b) must have different signs

Cython loop_example:
  Result: 0.0
  f(root) = 1.0
  BUG: Returned non-root without error!

Cython full_output_example:
  error_num: -1
  root: 0.0
  iterations: 1350451648
  Note: full_output correctly sets error_num=-1 (sign error)
```
</details>

## Why This Is A Bug

This bug violates the fundamental mathematical contract of bracketing root-finding algorithms. These methods (bisect, ridder, brenth, brentq) require that f(xa) and f(xb) have opposite signs to guarantee a root exists in the interval [xa, xb] by the Intermediate Value Theorem.

The specific issues are:

1. **Mathematical Incorrectness**: The function returns xa (0.0) as a "root" when f(0.0) = 1.0, which is clearly not a zero of the function.

2. **API Inconsistency**: The Python implementations (`scipy.optimize.bisect`, etc.) correctly raise a `ValueError` with the message "f(a) and f(b) must have different signs" for the same inputs.

3. **Silent Failure**: No exception or warning is raised, leading to silent propagation of incorrect results in numerical computations.

4. **Internal Inconsistency**: The `full_output_example` function correctly identifies the error condition (setting `error_num=-1`), but `loop_example` ignores this error status.

5. **Affects All Methods**: This bug affects all four bracketing methods available in the module.

## Relevant Context

The `loop_example` function is designed to demonstrate how to use Cython-wrapped optimization functions with mapped inputs. It's part of the scipy.optimize.cython_optimize module which provides Cython interfaces to underlying C optimization routines.

The underlying C functions correctly detect when f(xa) and f(xb) have the same sign and return an error code (-1). The bug is in the Python/Cython wrapper layer that fails to check this error code and raise an appropriate exception.

This is particularly concerning for scientific computing applications where silent numerical errors can propagate through complex calculations, potentially invalidating entire analyses.

Documentation: https://docs.scipy.org/doc/scipy/reference/optimize.cython_optimize.html

## Proposed Fix

Since the source code is in Cython and compiled, the exact fix would need to be applied to the Cython source file. The wrapper should check the error_num field from the underlying C function and raise appropriate exceptions:

```diff
# Pseudo-code for the fix in the Cython wrapper
def loop_example(method, a0_values, args, xa, xb, xtol, rtol, mitr):
    results = []
    for a0 in a0_values:
        full_output = call_underlying_solver(method, a0, args, xa, xb, xtol, rtol, mitr)
+       if full_output.error_num == -1:
+           raise ValueError("f(a) and f(b) must have different signs")
+       elif full_output.error_num == -2:
+           raise RuntimeError(f"Failed to converge after {mitr} iterations")
        results.append(full_output.root)
    return results
```

The actual implementation would need to be in Cython syntax, checking the `zeros_full_output` struct's `error_num` field and raising Python exceptions accordingly before appending the root to the results list.