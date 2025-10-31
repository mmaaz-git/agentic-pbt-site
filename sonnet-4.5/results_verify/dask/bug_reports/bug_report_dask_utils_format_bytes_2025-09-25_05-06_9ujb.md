# Bug Report: dask.utils.format_bytes Output Length Exceeds Documented Maximum

**Target**: `dask.utils.format_bytes`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function violates its documented invariant that output is "always <= 10 characters" for values < 2**60. Values >= 1000 PiB produce 11-character output.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
def test_format_bytes_output_length_invariant(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)} > 10"
```

**Failing input**: `1125894277343089729`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

from dask.utils import format_bytes

n = 1125894277343089729
result = format_bytes(n)

assert n < 2**60
assert len(result) == 11
assert result == '1000.00 PiB'
```

## Why This Is A Bug

The docstring explicitly states: "For all values < 2**60, the output is always <= 10 characters."

However, `format_bytes(1125894277343089729)` returns `'1000.00 PiB'` which has 11 characters. This violates the documented contract.

The issue occurs when values reach 1000 or more of any unit (e.g., "1000.00 PiB" has 11 chars vs "999.00 PiB" with 10 chars).

## Fix

Reduce decimal places dynamically to ensure output never exceeds 10 characters:

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1234,7 +1234,11 @@ def format_bytes(n: int) -> str:
         ("Mi", 2**20),
         ("ki", 2**10),
     ):
         if n >= k * 0.9:
-            return f"{n / k:.2f} {prefix}B"
+            if n >= k * 100:
+                return f"{n / k:.0f} {prefix}B"
+            elif n >= k * 10:
+                return f"{n / k:.1f} {prefix}B"
+            else:
+                return f"{n / k:.2f} {prefix}B"
     return f"{n} B"
```