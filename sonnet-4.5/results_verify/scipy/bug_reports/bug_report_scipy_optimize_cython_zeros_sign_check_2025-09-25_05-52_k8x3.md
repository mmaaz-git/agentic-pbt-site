# Bug Report: scipy.optimize.cython_optimize._zeros Missing Sign Check

**Target**: `scipy.optimize.cython_optimize._zeros.loop_example`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Cython root-finding functions (`bisect`, `brentq`, `brenth`, `ridder`) do not validate that `f(a)` and `f(b)` have opposite signs. When called with a bracket where both function values have the same sign, they silently return an invalid result (0.0) instead of raising a ValueError as the Python API does.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from scipy.optimize.cython_optimize import _zeros

@given(
    st.floats(min_value=-5.0, max_value=-0.1, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_no_sign_change_should_error(a0, offset):
    true_root = (-a0) ** (1.0/3.0)

    xa = true_root + offset
    xb = true_root + offset + 1.0

    f_xa = xa**3 + a0
    f_xb = xb**3 + a0

    assume(f_xa > 0 and f_xb > 0)

    methods = ['bisect', 'brentq', 'brenth', 'ridder']

    for method in methods:
        results = list(_zeros.loop_example(method, (a0,), (0.0, 0.0, 1.0), xa, xb, 0.01, 0.01, 50))

        assert len(results) == 0 or results[0] != 0.0, \
            f"{method} should error or return valid result when no sign change"
```

**Failing input**: `a0=-1.0, offset=1.0` (xa=2.0, xb=3.0, both f(xa)=7.0 and f(xb)=26.0 are positive)

## Reproducing the Bug

```python
from scipy.optimize.cython_optimize import _zeros
import scipy.optimize

a0 = -1.0
xa = 2.0
xb = 3.0

def f(x):
    return x**3 + a0

print(f"f(xa) = {f(xa)}")
print(f"f(xb) = {f(xb)}")
print(f"Both positive - no sign change!\n")

print("scipy.optimize.bisect (Python API):")
try:
    root = scipy.optimize.bisect(f, xa, xb)
except ValueError as e:
    print(f"  Correctly raises ValueError: {e}\n")

print("_zeros.loop_example (Cython API):")
for method in ['bisect', 'brentq', 'brenth', 'ridder']:
    results = list(_zeros.loop_example(method, (a0,), (0.0, 0.0, 1.0), xa, xb, 0.01, 0.01, 50))
    print(f"  {method}: returns {results}")
    if results:
        print(f"    root={results[0]} is in [{xa}, {xb}]? {xa <= results[0] <= xb}")
        print(f"    f(root)={f(results[0])} (should be ≈0)")
```

Output:
```
f(xa) = 7.0
f(xb) = 26.0
Both positive - no sign change!

scipy.optimize.bisect (Python API):
  Correctly raises ValueError: f(a) and f(b) must have different signs

_zeros.loop_example (Cython API):
  bisect: returns [0.0]
    root=0.0 is in [2.0, 3.0]? False
    f(root)=-1.0 (should be ≈0)
  brentq: returns [0.0]
    root=0.0 is in [2.0, 3.0]? False
    f(root)=-1.0 (should be ≈0)
  brenth: returns [0.0]
    root=0.0 is in [2.0, 3.0]? False
    f(root)=-1.0 (should be ≈0)
  ridder: returns [0.0]
    root=0.0 is in [2.0, 3.0]? False
    f(root)=-1.0 (should be ≈0)
```

## Why This Is A Bug

1. **Violates documented precondition**: All these root-finding methods require that f(a) and f(b) have opposite signs. The Python API enforces this by raising ValueError.

2. **Returns invalid result**: Instead of raising an error, the Cython functions return 0.0, which:
   - Is not in the search bracket [2.0, 3.0]
   - Is not a root (f(0.0) = -1.0, not 0)
   - Gives users no indication that their input was invalid

3. **Silent failure**: Users get a seemingly valid result that is actually meaningless, which can lead to incorrect downstream computations.

## Fix

The Cython implementation needs to add sign checking before proceeding with the root-finding algorithm. The fix should be added at the C level in the underlying root-finding functions (`bisect`, `brentq`, `brenth`, `ridder`) to validate that `f(xa) * f(xb) < 0` and return an appropriate error code if not.

Since the bug is in compiled Cython/C code and the source is not easily accessible in the installed package, a high-level fix would involve:

1. Adding sign validation in the C implementation of each method
2. Returning an error code (e.g., `error_num = -1` for sign error) via the `zeros_full_output` struct
3. Having the Python wrapper check this error code and either raise ValueError or return an empty result