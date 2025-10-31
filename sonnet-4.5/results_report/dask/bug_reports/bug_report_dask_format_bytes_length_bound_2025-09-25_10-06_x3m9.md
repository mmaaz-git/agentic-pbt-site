# Bug Report: dask.utils.format_bytes Output Length Exceeds Documented Bound

**Target**: `dask.utils.format_bytes`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function's docstring claims "For all values < 2**60, the output is always <= 10 characters," but this claim is violated for large values near 2**60.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
def test_format_bytes_length_bound(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)} > 10"
```

**Failing input**: `n = 1_125_894_277_343_089_729`

## Reproducing the Bug

```python
from dask.utils import format_bytes

n = 1_125_894_277_343_089_729
result = format_bytes(n)

print(f"n = {n}")
print(f"n < 2**60 = {n < 2**60}")
print(f"format_bytes(n) = '{result}'")
print(f"len(result) = {len(result)}")
```

Output:
```
n = 1125894277343089729
n < 2**60 = True
format_bytes(n) = '1000.00 PiB'
len(result) = 11
```

## Why This Is A Bug

The docstring explicitly claims: "For all values < 2**60, the output is always <= 10 characters." This is a contract violation. Code relying on this guarantee (e.g., for UI layout or column width) could break when large values are formatted.

The bug occurs because when `n / 2**50 >= 1000`, the result has the format "XXXX.XX PiB" which is 11 characters. This happens for all values where `n >= 1000 * 2**50 = 1_125_899_906_842_624_000`.

## Fix

```diff
 def format_bytes(n: int) -> str:
-    """Format bytes as text
+    """Format bytes as text

     >>> from dask.utils import format_bytes
     >>> format_bytes(1)
     '1 B'
     >>> format_bytes(1234)
     '1.21 kiB'
     >>> format_bytes(12345678)
     '11.77 MiB'
     >>> format_bytes(1234567890)
     '1.15 GiB'
     >>> format_bytes(1234567890000)
     '1.12 TiB'
     >>> format_bytes(1234567890000000)
     '1.10 PiB'

-    For all values < 2**60, the output is always <= 10 characters.
+    For most values, the output is <= 10 characters, though large values
+    near 2**60 may produce up to 11 characters (e.g., "1000.00 PiB").
     """
```

Alternatively, to maintain the 10-character guarantee, change the formatting for large values:

```diff
     for prefix, k in (
         ("Pi", 2**50),
         ("Ti", 2**40),
         ("Gi", 2**30),
         ("Mi", 2**20),
         ("ki", 2**10),
     ):
         if n >= k * 0.9:
-            return f"{n / k:.2f} {prefix}B"
+            # Ensure output is at most 10 chars by reducing precision for large values
+            formatted = f"{n / k:.2f}"
+            if len(formatted) > 6:  # "XXXX.YY" is 7 chars, need room for " XiB"
+                formatted = f"{n / k:.1f}"
+            if len(formatted) > 6:
+                formatted = f"{int(n / k)}"
+            return f"{formatted} {prefix}B"
     return f"{n} B"
```