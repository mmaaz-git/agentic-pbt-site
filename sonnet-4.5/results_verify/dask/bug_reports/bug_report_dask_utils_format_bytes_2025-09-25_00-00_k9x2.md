# Bug Report: dask.utils.format_bytes Output Length Exceeds Documented Limit

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function's docstring claims "For all values < 2**60, the output is always <= 10 characters," but this property is violated for large values near 2**60, where outputs can be 11 characters long.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.utils import format_bytes


@given(st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=1000)
def test_format_bytes_length_property(n):
    """For all values < 2**60, output should be <= 10 characters."""
    result = format_bytes(n)
    assert len(result) <= 10, f"Output exceeds 10 chars for {n}: '{result}' (len={len(result)})"
```

**Failing input**: `n=1125894277343089729`

## Reproducing the Bug

```python
from dask.utils import format_bytes

n = 1125894277343089729
result = format_bytes(n)

assert n < 2**60
assert len(result) > 10

print(f"format_bytes({n}) = '{result}'")
print(f"Length: {len(result)} (expected <= 10)")
```

Output:
```
format_bytes(1125894277343089729) = '1000.00 PiB'
Length: 11 (expected <= 10)
```

## Why This Is A Bug

The function's docstring at line 1788 explicitly states: "For all values < 2**60, the output is always <= 10 characters."

However, when formatting values >= 1000 PiB, the output becomes 11 characters (e.g., "1000.00 PiB" or "1024.00 PiB"), violating this documented guarantee.

## Fix

The issue occurs because values near 2**60 result in outputs like "1000.00 PiB" or "1024.00 PiB" (11 chars), when the threshold is k * 0.9 for each unit. The fix is to either:

1. **Update the docstring** to reflect the actual behavior (easiest fix):

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1785,7 +1785,7 @@ def format_bytes(n: int) -> str:
     >>> format_bytes(1234567890000000)
     '1.10 PiB'

-    For all values < 2**60, the output is always <= 10 characters.
+    For most values, the output is typically <= 10 characters.
     """
     for prefix, k in (
         ("Pi", 2**50),
```

2. **Or adjust the formatting** to guarantee the property by using fewer decimal places for large values, though this would reduce precision.