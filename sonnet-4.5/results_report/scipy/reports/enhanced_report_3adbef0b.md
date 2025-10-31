# Bug Report: scipy.differentiate.derivative Missing initial_step Validation

**Target**: `scipy.differentiate.derivative`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `derivative` function fails to validate that `initial_step` must be positive, instead silently converting non-positive values to NaN and proceeding with computation, resulting in opaque failure modes rather than raising a clear `ValueError`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
from scipy.differentiate import derivative
import numpy as np

def test_derivative_should_reject_non_positive_initial_step(initial_step):
    """Test that derivative raises ValueError for non-positive initial_step"""
    try:
        result = derivative(np.sin, 1.0, initial_step=initial_step)
        # If we got here, no error was raised - test failed
        raise AssertionError(f"Expected ValueError but got result: success={result.success}, status={result.status}, df={result.df}")
    except ValueError as e:
        if "initial_step" in str(e):
            # Good - got the expected error
            pass
        else:
            # Wrong error message
            raise AssertionError(f"Got ValueError but wrong message: {e}")

# Run the test
if __name__ == "__main__":
    # Run with a specific example first
    print("Testing with initial_step=0.0:")
    try:
        test_derivative_should_reject_non_positive_initial_step(0.0)
        print("Test passed - ValueError was raised as expected")
    except AssertionError as e:
        print(f"Test failed: {e}")

    print("\nTesting with initial_step=-1.0:")
    try:
        test_derivative_should_reject_non_positive_initial_step(-1.0)
        print("Test passed - ValueError was raised as expected")
    except AssertionError as e:
        print(f"Test failed: {e}")

    print("\nRunning property-based test with Hypothesis...")
    from hypothesis import given

    @given(st.floats(min_value=-10, max_value=0, allow_nan=False, allow_infinity=False))
    @example(0.0)
    @example(-1.0)
    @example(-0.5)
    def test_with_hypothesis(initial_step):
        test_derivative_should_reject_non_positive_initial_step(initial_step)

    try:
        test_with_hypothesis()
        print("All Hypothesis tests passed - ValueError was raised as expected")
    except AssertionError as e:
        print(f"Hypothesis test failed: {e}")
        print("This demonstrates that the function does NOT raise ValueError for non-positive initial_step values")
```

<details>

<summary>
**Failing input**: `initial_step=0.0`
</summary>
```
Testing with initial_step=0.0:
Test failed: Expected ValueError but got result: success=False, status=-3, df=nan

Testing with initial_step=-1.0:
Test failed: Expected ValueError but got result: success=False, status=-3, df=nan

Running property-based test with Hypothesis...
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 47, in <module>
  |     test_with_hypothesis()
  |     ~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 40, in test_with_hypothesis
  |     @example(0.0)
  |                ^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 3 distinct failures in explicit examples. (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 44, in test_with_hypothesis
    |     test_derivative_should_reject_non_positive_initial_step(initial_step)
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
    |   File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 10, in test_derivative_should_reject_non_positive_initial_step
    |     raise AssertionError(f"Expected ValueError but got result: success={result.success}, status={result.status}, df={result.df}")
    | AssertionError: Expected ValueError but got result: success=False, status=-3, df=nan
    | Falsifying explicit example: test_with_hypothesis(
    |     initial_step=0.0,
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 44, in test_with_hypothesis
    |     test_derivative_should_reject_non_positive_initial_step(initial_step)
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
    |   File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 10, in test_derivative_should_reject_non_positive_initial_step
    |     raise AssertionError(f"Expected ValueError but got result: success={result.success}, status={result.status}, df={result.df}")
    | AssertionError: Expected ValueError but got result: success=False, status=-3, df=nan
    | Falsifying explicit example: test_with_hypothesis(
    |     initial_step=-1.0,
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 44, in test_with_hypothesis
    |     test_derivative_should_reject_non_positive_initial_step(initial_step)
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
    |   File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 10, in test_derivative_should_reject_non_positive_initial_step
    |     raise AssertionError(f"Expected ValueError but got result: success={result.success}, status={result.status}, df={result.df}")
    | AssertionError: Expected ValueError but got result: success=False, status=-3, df=nan
    | Falsifying explicit example: test_with_hypothesis(
    |     initial_step=-0.5,
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.differentiate import derivative

# Test with initial_step=0.0
result = derivative(np.sin, 1.0, initial_step=0.0)
print("Testing with initial_step=0.0:")
print(f"  success: {result.success}")
print(f"  status: {result.status}")
print(f"  df: {result.df}")
print(f"  error: {result.error}")
print(f"  nfev: {result.nfev}")
print(f"  nit: {result.nit}")
print()

# Test with initial_step=-1.0 (negative)
result = derivative(np.sin, 1.0, initial_step=-1.0)
print("Testing with initial_step=-1.0:")
print(f"  success: {result.success}")
print(f"  status: {result.status}")
print(f"  df: {result.df}")
print(f"  error: {result.error}")
print(f"  nfev: {result.nfev}")
print(f"  nit: {result.nit}")
print()

# For comparison, test with valid initial_step=1.0
result = derivative(np.sin, 1.0, initial_step=1.0)
print("Testing with initial_step=1.0 (valid):")
print(f"  success: {result.success}")
print(f"  status: {result.status}")
print(f"  df: {result.df}")
print(f"  error: {result.error}")
print(f"  nfev: {result.nfev}")
print(f"  nit: {result.nit}")
```

<details>

<summary>
Output demonstrating the silent failure instead of ValueError
</summary>
```
Testing with initial_step=0.0:
  success: False
  status: -3
  df: nan
  error: nan
  nfev: 9
  nit: 1

Testing with initial_step=-1.0:
  success: False
  status: -3
  df: nan
  error: nan
  nfev: 9
  nit: 1

Testing with initial_step=1.0 (valid):
  success: True
  status: 0
  df: 0.5403023058667242
  error: 3.57734397660181e-10
  nfev: 11
  nit: 2
```
</details>

## Why This Is A Bug

The function currently accepts non-positive `initial_step` values without raising a `ValueError` during input validation, instead silently converting them to NaN at line 408 and allowing computation to proceed until it fails with a cryptic `status=-3` (non-finite value encountered). This violates several principles and expectations:

1. **Inconsistent validation**: The error message at line 27 of `_differentiate.py` explicitly states "Tolerances and step parameters must be non-negative scalars." The `step_factor` parameter is validated against this requirement (lines 28-34), but `initial_step` - which is also a step parameter - is not validated at all in the `_derivative_iv` function.

2. **Silent failure**: Line 408 explicitly handles non-positive step sizes by converting them to NaN: `h0 = xpx.at(h0)[h0 <= 0].set(xp.nan)`. This shows the developers knew non-positive values were invalid, but chose to handle them silently rather than failing fast with a clear error message.

3. **Mathematical requirement**: Finite difference formulas require non-zero step sizes to be well-defined (division by zero). A step size of 0 is mathematically meaningless.

4. **Poor user experience**: Users who accidentally pass `initial_step=0` or negative values get a confusing `success=False` with `status=-3` and `df=nan`, rather than a clear `ValueError` explaining that `initial_step` must be positive.

5. **Documentation gap**: While the documentation describes `initial_step` as "The (absolute) initial step size", it doesn't explicitly state it must be positive, leaving users to discover this requirement through trial and error.

## Relevant Context

The validation function `_derivative_iv` (lines 11-57) is responsible for input validation. It currently validates:
- `f` is callable (line 16-17)
- Tolerances and `step_factor` are non-negative scalars (lines 27-34)
- `maxiter` is a positive integer (lines 36-38)
- `order` is a positive integer (lines 40-42)
- `preserve_shape` is boolean (lines 49-51)

However, it fails to validate `initial_step`, even though the error message mentions "step parameters" (plural) and the code later treats non-positive values as invalid.

The fix is straightforward - add validation for `initial_step` after it's converted to an array but before returning from the validation function.

## Proposed Fix

```diff
diff --git a/scipy/differentiate/_differentiate.py b/scipy/differentiate/_differentiate.py
index abc123..def456 100644
--- a/scipy/differentiate/_differentiate.py
+++ b/scipy/differentiate/_differentiate.py
@@ -44,6 +44,14 @@ def _derivative_iv(f, x, args, tolerances, maxiter, order, initial_step,
     step_direction = xp.asarray(step_direction)
     initial_step = xp.asarray(initial_step)
     temp = xp.broadcast_arrays(x, step_direction, initial_step)
     x, step_direction, initial_step = temp
+
+    # Validate that initial_step is positive
+    # Using NumPy for validation since initial_step might not support all array operations yet
+    initial_step_np = np.asarray(initial_step)
+    if np.any(initial_step_np <= 0) or np.any(np.isnan(initial_step_np)):
+        raise ValueError('`initial_step` must contain only positive values.')

     message = '`preserve_shape` must be True or False.'
     if preserve_shape not in {True, False}:
```