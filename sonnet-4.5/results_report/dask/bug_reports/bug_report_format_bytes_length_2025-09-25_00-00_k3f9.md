# Bug Report: format_bytes Length Guarantee Violated

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function's docstring claims "For all values < 2**60, the output is always <= 10 characters", but this is violated for values >= 1000 PiB.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.utils import format_bytes


@given(st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=1000)
def test_format_bytes_length_claim(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = {result!r} has length {len(result)}, expected <= 10"
```

**Failing input**: `n=1_125_894_277_343_089_729`

## Reproducing the Bug

```python
from dask.utils import format_bytes

n = 1_125_894_277_343_089_729
result = format_bytes(n)

print(f"Input: {n}")
print(f"Input < 2**60: {n < 2**60}")
print(f"Output: {result!r}")
print(f"Output length: {len(result)}")

assert len(result) <= 10, f"Expected <= 10 characters, got {len(result)}"
```

Output:
```
Input: 1125894277343089729
Input < 2**60: True
Output: '1000.00 PiB'
Output length: 11
AssertionError: Expected <= 10 characters, got 11
```

Additional failing cases:
- `format_bytes(1125899906842624000)` → `'1000.00 PiB'` (11 chars)
- `format_bytes(1152921504606846975)` → `'1024.00 PiB'` (11 chars)

## Why This Is A Bug

The function's docstring explicitly states: "For all values < 2**60, the output is always <= 10 characters." This is a documented guarantee that users may rely on for formatting purposes (e.g., fixed-width displays). The implementation violates this contract for values >= 1000 PiB.

## Fix

The issue occurs because values >= 1000 PiB use the format `f"{n / k:.2f} {prefix}B"`, which produces 11-character strings. One fix is to reduce decimal precision for large values:

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1,7 +1,11 @@
 def format_bytes(n: int) -> str:
     for prefix, k in (
         ("Pi", 2**50),
         ("Ti", 2**40),
         ("Gi", 2**30),
         ("Mi", 2**20),
         ("ki", 2**10),
     ):
         if n >= k * 0.9:
-            return f"{n / k:.2f} {prefix}B"
+            value = n / k
+            if value >= 1000:
+                return f"{value:.1f} {prefix}B"
+            else:
+                return f"{value:.2f} {prefix}B"
     return f"{n} B"
```

This changes outputs like "1000.00 PiB" to "1000.0 PiB" (10 chars), maintaining the 10-character guarantee.