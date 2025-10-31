# Bug Report: dask.utils.format_bytes Length Constraint Violation

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function violates its documented constraint that "For all values < 2**60, the output is always <= 10 characters" when formatting values >= 1000 PiB.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.utils import format_bytes

@settings(max_examples=1000)
@given(st.integers(min_value=0, max_value=2**60-1))
def test_format_bytes_length_constraint_documented(n):
    """Property: format_bytes should output <= 10 chars for n < 2**60 (documented claim)"""
    result = format_bytes(n)

    if n < 2**60:
        assert len(result) <= 10, f"VIOLATION: format_bytes({n}) = '{result}' (length={len(result)} > 10)"
```

**Failing input**: `1125899906842624000` (exactly 1000 PiB)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.utils import format_bytes

n = 1000 * 2**50
result = format_bytes(n)

print(f"Value: {n}")
print(f"Output: '{result}'")
print(f"Length: {len(result)}")
print(f"2**60: {2**60}")
print(f"Value < 2**60: {n < 2**60}")
print(f"Length <= 10: {len(result) <= 10}")
```

## Why This Is A Bug

The function's docstring explicitly states: "For all values < 2**60, the output is always <= 10 characters."

However, for byte values between 1000 PiB and 1024 PiB (which is 2**60), the function outputs strings like "1000.00 PiB" which are 11 characters long, violating this documented constraint.

Examples of violations:
- `format_bytes(1000 * 2**50)` → `"1000.00 PiB"` (11 chars)
- `format_bytes(1023 * 2**50)` → `"1023.00 PiB"` (11 chars)
- `format_bytes(2**60 - 1)` → `"1024.00 PiB"` (11 chars)

## Fix

The issue is that four-digit values (1000-1023) with ".00" and a 4-character unit (" PiB") exceed 10 characters. The fix should either:

1. Update the documentation to reflect the actual behavior (11 characters for values >= 1000 PiB)
2. Change the formatting to ensure the constraint holds (e.g., reduce decimal places for large values)

Option 1 (documentation fix):

```diff
 def format_bytes(n: int) -> str:
     """Format bytes as text

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
+    For values < 1000 PiB, the output is always <= 10 characters.
+    For values >= 1000 PiB and < 2**60, the output may be up to 11 characters.
     """
```

Option 2 (code fix to maintain 10-character constraint):

```diff
 def format_bytes(n: int) -> str:
     """Format bytes as text

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

     For all values < 2**60, the output is always <= 10 characters.
     """
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