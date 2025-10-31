# Bug Report: scipy.optimize Root Finders Return Uninitialized Memory for Iterations Count When Root is at Boundary

**Target**: `scipy.optimize.{bisect, ridder, brenth, brentq}`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When a root is found exactly at one of the bracketing interval boundaries (where f(a)=0 or f(b)=0), all four root-finding functions return uninitialized memory values in the `iterations` field of the RootResults object instead of a valid iteration count.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from scipy.optimize import bisect, ridder, brenth, brentq


@given(
    root=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    offset=st.floats(min_value=0.1, max_value=50, allow_nan=False, allow_infinity=False),
)
def test_iterations_field_boundary_root(root, offset):
    assume(abs(root) < 100)
    assume(offset > 0.1)

    def f(x):
        return x - root

    a = root - offset
    b = root

    for method in [bisect, ridder, brenth, brentq]:
        root_val, info = method(f, a, b, full_output=True)

        assert isinstance(info.iterations, int)
        assert 0 <= info.iterations <= 1000, \
            f"{method.__name__}: iterations = {info.iterations} (should be small non-negative int)"


if __name__ == "__main__":
    test_iterations_field_boundary_root()
```

<details>

<summary>
**Failing input**: `root=0.0, offset=1.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 28, in <module>
    test_iterations_field_boundary_root()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 6, in test_iterations_field_boundary_root
    root=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
              ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 23, in test_iterations_field_boundary_root
    assert 0 <= info.iterations <= 1000, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: bisect: iterations = -1975240512 (should be small non-negative int)
Falsifying example: test_iterations_field_boundary_root(
    # The test always failed when commented parts were varied together.
    root=0.0,  # or any other generated value
    offset=1.0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from scipy.optimize import bisect, ridder, brenth, brentq


def f(x):
    return x - 5

a = 0.0
b = 5.0

print("Testing boundary root (root at b=5):")
print("=" * 50)
for method in [bisect, ridder, brenth, brentq]:
    root, info = method(f, a, b, full_output=True)
    print(f"{method.__name__}:")
    print(f"  root = {root}")
    print(f"  iterations = {info.iterations}")
    print(f"  function_calls = {info.function_calls}")
    print(f"  converged = {info.converged}")
    print()
```

<details>

<summary>
Output showing uninitialized memory values in iterations field
</summary>
```
Testing boundary root (root at b=5):
==================================================
bisect:
  root = 5.0
  iterations = 6050768
  function_calls = 2
  converged = True

ridder:
  root = 5.0
  iterations = 6050768
  function_calls = 2
  converged = True

brenth:
  root = 5.0
  iterations = -1418546688
  function_calls = 2
  converged = True

brentq:
  root = 5.0
  iterations = -1418546688
  function_calls = 2
  converged = True

```
</details>

## Why This Is A Bug

This violates the documented contract of the `RootResults.iterations` field, which according to scipy documentation should be an integer representing "Number of iterations needed to find the root". The actual behavior returns uninitialized memory values (e.g., 6050768, -1418546688) instead of meaningful iteration counts.

The bug specifically manifests when:
1. The root is exactly at one of the interval boundaries (f(a)=0 or f(b)=0)
2. The C/Cython implementation takes an early return path after evaluating the boundary values
3. The `iterations` field in the output structure is never initialized before being returned

This is a serious issue because:
- **Contract violation**: The documented API contract guarantees the iterations field contains the "Number of iterations needed to find the root", not garbage values
- **Non-deterministic behavior**: Uninitialized memory values can vary between runs, making debugging difficult
- **Potential security risk**: Returning uninitialized memory could theoretically leak sensitive information
- **Breaks analysis tools**: Code that relies on the iterations field for performance analysis or algorithm comparison will receive meaningless data

The functions correctly identify the root and set other fields properly (`root`, `function_calls`, `converged`), indicating that boundary roots are a supported use case with an implementation bug rather than undefined behavior.

## Relevant Context

The scipy.optimize module provides several root-finding algorithms that use bracketing methods. These functions share similar C/Cython implementations that check boundary conditions early in execution. When `f(a)=0` or `f(b)=0`, the implementation correctly identifies this as a root and returns early, but fails to initialize the `iterations` field in the `RootResults` structure.

The documentation states that `f(a)` and `f(b)` must have opposite signs, which technically allows for one to be zero (since zero is considered to have opposite sign to both positive and negative values in this context). The fact that the functions handle boundary roots correctly in all other respects confirms this is intended behavior.

For reference:
- scipy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.RootResults.html
- The bug affects scipy.optimize.bisect, scipy.optimize.ridder, scipy.optimize.brenth, and scipy.optimize.brentq

## Proposed Fix

The fix requires initializing the `iterations` field to 0 in the C/Cython implementation before checking boundary conditions. Since the actual C source is compiled, here's the conceptual fix:

```diff
In scipy/optimize/Zeros/bisect.c, ridder.c, brenth.c, brentq.c:

  zeros_full_output result;
+ result.iterations = 0;  // Initialize iterations counter
+ result.funcalls = 0;    // Initialize function call counter

  double fa = f(a);
+ result.funcalls = 1;
  if (fa == 0.0) {
      result.root = a;
-     // iterations not set - BUG!
+     result.iterations = 0;  // No iterations needed for boundary root
      result.converged = 1;
      return result;
  }

  double fb = f(b);
+ result.funcalls = 2;
  if (fb == 0.0) {
      result.root = b;
-     // iterations not set - BUG!
+     result.iterations = 0;  // No iterations needed for boundary root
      result.converged = 1;
      return result;
  }
```