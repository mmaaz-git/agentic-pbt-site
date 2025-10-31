# Bug Report: format_bytes Length Constraint Violation

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function's docstring claims "For all values < 2**60, the output is always <= 10 characters", but this claim is violated for values near the PiB boundary, where the output can be 11 characters.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60))
def test_format_bytes_length_constraint(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)} > 10"
```

**Failing input**: `n = 1125894277343089729` (approximately 1000 PiB)

## Reproducing the Bug

```python
from dask.utils import format_bytes

n = 1125894277343089729
result = format_bytes(n)
print(f"format_bytes({n}) = '{result}'")
print(f"Length: {len(result)}")
print(f"n < 2**60: {n < 2**60}")
```

Output:
```
format_bytes(1125894277343089729) = '1000.00 PiB'
Length: 11
n < 2**60: True
```

## Why This Is A Bug

The function's docstring explicitly states:

> For all values < 2**60, the output is always <= 10 characters.

However, values >= 1000 * 2**50 (which is less than 2**60) produce outputs like "1000.00 PiB" which have 11 characters, violating this documented constraint.

## Fix

The issue occurs when formatting values >= 1000 * 2**50. The function uses `.2f` formatting which always produces 2 decimal places. For values >= 1000, this results in at least 7 digits (`1000.00`), plus the space and 3-character unit (`PiB`), totaling 11 characters.

Possible fixes:
1. Update the docstring to accurately reflect the actual maximum length (11 characters for values < 2**60)
2. Modify the formatting logic to reduce decimal places when the value is >= 100 to maintain the 10-character guarantee
3. Change the threshold in the docstring from 2**60 to a value where the constraint actually holds (e.g., 1000 * 2**50)

The simplest fix is to update the docstring:

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -XXX,7 +XXX,7 @@ def format_bytes(n: int) -> str:
     >>> format_bytes(1234567890000000)
     '1.10 PiB'

-    For all values < 2**60, the output is always <= 10 characters.
+    For all values < 2**60, the output is always <= 11 characters.
     """
     for prefix, k in (
         ("Pi", 2**50),
```