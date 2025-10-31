# Bug Report: dask.widgets format_bytes violates length guarantee

**Target**: `dask.utils.format_bytes` (exposed via `dask.widgets.FILTERS['format_bytes']`)
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function's docstring claims "For all values < 2**60, the output is always <= 10 characters", but this property is violated for large values approaching 2**60.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.widgets import FILTERS


@given(st.integers(min_value=0, max_value=2**60 - 1))
def test_format_bytes_output_length(n):
    format_bytes = FILTERS['format_bytes']
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)} > 10"
```

**Failing input**: `n = 1125894277343089729`

## Reproducing the Bug

```python
from dask.utils import format_bytes

n = 1125894277343089729
result = format_bytes(n)

print(f"format_bytes({n}) = '{result}'")
print(f"Length: {len(result)}")
print(f"n < 2**60: {n < 2**60}")

assert n < 2**60
assert len(result) > 10
```

Output:
```
format_bytes(1125894277343089729) = '1000.00 PiB'
Length: 11
n < 2**60: True
```

## Why This Is A Bug

The docstring explicitly states: "For all values < 2**60, the output is always <= 10 characters."

However, when `n >= 1000 * 2**50` (approximately `1.126e18`), the output uses PiB formatting and produces strings like "1000.00 PiB" (11 characters) or larger, violating the documented guarantee.

The root cause is that the function formats with `.2f` precision, which for values >= 1000 PiB produces 4+ digits before the decimal point, resulting in 11+ character strings.

## Fix

The fix requires either:
1. Updating the docstring to reflect the actual behavior (e.g., "For all values < 1000 * 2**50, the output is always <= 10 characters")
2. Changing the formatting logic to ensure the 10-character limit is maintained (e.g., reducing precision or switching to scientific notation for very large values)

Option 1 (documentation fix):

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1785,7 +1785,7 @@ def format_bytes(n: int) -> str:
     >>> format_bytes(1234567890000000)
     '1.10 PiB'

-    For all values < 2**60, the output is always <= 10 characters.
+    For most typical values, the output is <= 10 characters.
     """
     for prefix, k in (
         ("Pi", 2**50),
```