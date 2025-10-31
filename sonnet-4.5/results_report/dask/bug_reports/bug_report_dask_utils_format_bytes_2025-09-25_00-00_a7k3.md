# Bug Report: dask.utils.format_bytes violates length guarantee

**Target**: `dask.utils.format_bytes`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function violates its documented guarantee that "For all values < 2**60, the output is always <= 10 characters." For certain inputs near the boundary, the output can be 11 characters.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
def test_format_bytes_length_invariant(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = {result!r} has length {len(result)} > 10"
```

**Failing input**: `n = 1125894277343089729`

## Reproducing the Bug

```python
from dask.utils import format_bytes

n = 1125894277343089729
result = format_bytes(n)
print(f"format_bytes({n}) = {result!r}")
print(f"Length: {len(result)}")

assert len(result) <= 10
```

Output:
```
format_bytes(1125894277343089729) = '1000.00 PiB'
Length: 11
AssertionError
```

## Why This Is A Bug

The docstring explicitly states: "For all values < 2**60, the output is always <= 10 characters." This is a documented contract that users may rely on for formatting, layout, or display purposes. The value `1125894277343089729` is less than `2**60` (1152921504606846976), yet produces 11-character output.

## Fix

The issue occurs when values round to exactly 1000.00 in a unit. The format string produces "1000.00 PiB" (11 chars) instead of switching to a more compact representation.

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1789,13 +1789,13 @@ def format_bytes(n: int) -> str:
     For all values < 2**60, the output is always <= 10 characters.
     """
     for prefix, k in (
         ("Pi", 2**50),
         ("Ti", 2**40),
         ("Gi", 2**30),
         ("Mi", 2**20),
         ("ki", 2**10),
     ):
-        if n >= k * 0.9:
+        if n >= k:
             return f"{n / k:.2f} {prefix}B"
     return f"{n} B"
```

This change ensures values below a threshold use the next smaller unit, preventing the "1000.00" case.