# Bug Report: dask.dataframe.dask_expr.memory_repr Returns None for Large Values

**Target**: `dask.dataframe.dask_expr.memory_repr`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `memory_repr` function returns `None` instead of a string when the input value exceeds 1024 TB (1024^5 bytes), violating the function's implicit contract of always returning a string representation.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import dask.dataframe.dask_expr as de

@given(st.floats(min_value=0, max_value=1e20, allow_nan=False, allow_infinity=False))
def test_memory_repr_returns_string_with_unit(num):
    result = de.memory_repr(num)
    assert isinstance(result, str)
```

**Failing input**: `1125899906842624.0` (1 PB = 1024^5 bytes)

## Reproducing the Bug

```python
import dask.dataframe.dask_expr as de

result = de.memory_repr(1024**5)
print(f"Result: {result!r}")

assert result is not None, "memory_repr should never return None"
```

## Why This Is A Bug

The `memory_repr` function is designed to convert byte counts to human-readable format. However, when the value exceeds 1024 TB, the function exhausts all available units (bytes, KB, MB, GB, TB) without returning a value, resulting in an implicit `None` return. This violates the function's implicit contract and would cause issues in any code that uses this function for display or logging purposes, expecting a string result.

## Fix

```diff
def memory_repr(num):
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:3.1f} {x}"
        num /= 1024.0
+   return f"{num:3.1f} PB"
```

Alternatively, the function could continue with more units (PB, EB, ZB, YB) or fall back to scientific notation for extremely large values.