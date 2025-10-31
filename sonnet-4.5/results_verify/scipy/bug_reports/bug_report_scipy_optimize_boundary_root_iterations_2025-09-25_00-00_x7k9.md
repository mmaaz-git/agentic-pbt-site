# Bug Report: scipy.optimize root finders return garbage iterations count for boundary roots

**Target**: `scipy.optimize.{bisect, ridder, brenth, brentq}`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When a root is found exactly at one of the bracketing interval boundaries (either `a` or `b`), the `iterations` field in the returned `RootResults` object contains uninitialized garbage values instead of the actual iteration count.

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
```

**Failing input**: `root=0.0, offset=1.0` (or any values where root is at boundary)

## Reproducing the Bug

```python
from scipy.optimize import bisect, ridder, brenth, brentq


def f(x):
    return x - 5

a = 0.0
b = 5.0

for method in [bisect, ridder, brenth, brentq]:
    root, info = method(f, a, b, full_output=True)
    print(f"{method.__name__}:")
    print(f"  root = {root}")
    print(f"  iterations = {info.iterations}")
    print(f"  function_calls = {info.function_calls}")
    print()
```

**Output:**
```
bisect:
  root = 5.0
  iterations = 6051019
  function_calls = 2

ridder:
  root = 5.0
  iterations = 6051126
  function_calls = 2

brenth:
  root = 5.0
  iterations = 505590272
  function_calls = 2

brentq:
  root = 5.0
  iterations = 505590272
  function_calls = 2
```

The `iterations` field contains garbage values like 6051019, 505590272 instead of the expected small values (should be 0 or 1).

## Why This Is A Bug

1. **Contract violation**: The `RootResults.iterations` field is documented as "Number of iterations needed to find the root" and should always be a small non-negative integer. Returning uninitialized memory violates this contract.

2. **Unreliable information**: Users relying on this field for performance analysis or debugging will get misleading garbage data.

3. **Determinism**: The garbage values are non-deterministic (depend on uninitialized memory), making debugging harder.

## Fix

The bug occurs in the underlying C/Cython implementation when the root is found during the initial function evaluations at the boundaries. The `iterations` variable is not being initialized before being returned in this early-exit code path.

The fix should initialize the `iterations` field to 0 (or appropriate value) before checking if `f(a) == 0` or `f(b) == 0` in the C implementation of `bisect`, `ridder`, `brenth`, and `brentq`.

Since the C source is not readily available in this environment, here's the conceptual fix:

```diff
In the C/Cython implementation of each method (bisect, ridder, brenth, brentq):

  cdef zeros_full_output full_output_struct
+ full_output_struct.iterations = 0  # Initialize iterations
+ full_output_struct.funcalls = 0    # Initialize funcalls

  fa = f(a)
+ full_output_struct.funcalls = 1
  if fa == 0:
      full_output_struct.root = a
-     # iterations not set here - BUG!
+     full_output_struct.iterations = 0
      return full_output_struct

  fb = f(b)
+ full_output_struct.funcalls = 2
  if fb == 0:
      full_output_struct.root = b
-     # iterations not set here - BUG!
+     full_output_struct.iterations = 0
      return full_output_struct
```

The fix ensures that all fields in the output structure are properly initialized before any early returns.