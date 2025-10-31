# Bug Report: scipy.optimize.cython_optimize Silent Failure on Invalid Inputs

**Target**: `scipy.optimize.cython_optimize._zeros`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Functions in EXAMPLES_MAP (brentq, bisect, ridder, brenth) silently swallow exceptions from invalid inputs and return garbage values (0.0) instead of raising errors. This causes silent failures that are difficult to debug.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import scipy.optimize.cython_optimize._zeros as zeros


@given(
    num_args=st.integers(min_value=0, max_value=10).filter(lambda x: x != 4)
)
def test_validates_arg_count(num_args):
    args = tuple([1.0] * num_args)

    try:
        result = zeros.EXAMPLES_MAP['brentq'](args, 1.0, 2.0, 1e-6, 1e-6, 100)
        assert False, f"Should have raised an error for {num_args} args, but returned {result}"
    except (IndexError, ValueError, TypeError):
        pass
```

**Failing input**: `num_args=1` (and any value except 4)

## Reproducing the Bug

```python
import scipy.optimize.cython_optimize._zeros as zeros

args_too_few = (1.0,)
result = zeros.EXAMPLES_MAP['brentq'](args_too_few, 1.0, 2.0, 1e-6, 1e-6, 100)

print(f"Result with 1 arg instead of 4: {result}")
```

Output:
```
Result with 1 arg instead of 4: 0.0
```

The function returns 0.0 instead of raising an IndexError. The exception is raised internally but swallowed:
```
IndexError: not enough values found during array assignment, expected 4, got 1
```

## Why This Is A Bug

1. **Silent failures**: Users get incorrect results (0.0) without any indication that their inputs were invalid
2. **Violation of Python conventions**: Functions should raise exceptions for invalid inputs, not return garbage
3. **Difficult to debug**: Users may not realize their code is broken until they notice incorrect results downstream
4. **Type safety**: Passing non-numeric values also returns 0.0 instead of raising TypeError

## Fix

The Cython code needs to properly propagate exceptions instead of swallowing them. In the wrapper functions, exception handling should be removed or modified to allow errors to bubble up to the caller.

Expected behavior:
```python
zeros.EXAMPLES_MAP['brentq']((1.0,), 1.0, 2.0, 1e-6, 1e-6, 100)

IndexError: not enough values found during array assignment, expected 4, got 1
```