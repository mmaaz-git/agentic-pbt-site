# Bug Report: pandas.compat.numpy.function.process_skipna Type Contract Violation

**Target**: `pandas.compat.numpy.function.process_skipna`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`process_skipna` violates its type annotation `tuple[bool, Any]` by returning `np.bool_` instead of Python `bool` when given `np.bool_` input, creating inconsistency with similar functions.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from pandas.compat.numpy.function import process_skipna


@given(
    skipna=st.one_of(st.booleans(), st.none(), st.from_type(np.ndarray)),
    args=st.tuples(st.integers(), st.text())
)
def test_process_skipna_returns_python_bool(skipna, args):
    result_skipna, result_args = process_skipna(skipna, args)
    assert isinstance(result_skipna, bool) and not isinstance(result_skipna, np.bool_)
```

**Failing input**: `skipna=np.bool_(True)` or `skipna=np.bool_(False)`

## Reproducing the Bug

```python
import numpy as np
from pandas.compat.numpy.function import process_skipna

np_true = np.bool_(True)
result_skipna, _ = process_skipna(np_true, ())

print(f"Return type: {type(result_skipna)}")
print(f"Is Python bool: {type(result_skipna) == bool}")
print(f"Is np.bool_: {isinstance(result_skipna, np.bool_)}")

assert isinstance(result_skipna, bool) and not isinstance(result_skipna, np.bool_)
```

## Why This Is A Bug

1. **Type annotation violation**: The function signature declares `-> tuple[bool, Any]`, but returns `tuple[np.bool_, Any]` when given `np.bool_` input
2. **Inconsistency with related functions**: `validate_cum_func_with_skipna` explicitly converts `np.bool_` to `bool` with `bool(skipna)`, demonstrating the codebase's intent to use Python bools
3. **Potential downstream issues**: Code expecting Python `bool` may behave unexpectedly with `np.bool_` (e.g., type checks, JSON serialization, mypy violations)

## Fix

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -115,6 +115,8 @@ def process_skipna(skipna: bool | ndarray | None, args) -> tuple[bool, Any]:
     if isinstance(skipna, ndarray) or skipna is None:
         args = (skipna,) + args
         skipna = True
+    elif isinstance(skipna, np.bool_):
+        skipna = bool(skipna)

     return skipna, args
```