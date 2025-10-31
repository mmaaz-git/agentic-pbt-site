# Bug Report: dask.utils.format_bytes Length Claim Violation

**Target**: `dask.utils.format_bytes`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The docstring of `format_bytes` claims "For all values < 2**60, the output is always <= 10 characters", but this is violated for values >= 1000 * 2**50 (which is < 2**60), where the output is 11 characters or more.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings


@given(st.integers(min_value=1, max_value=2**60))
@settings(max_examples=1000)
def test_format_bytes_length_claim(n):
    result = dask.utils.format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)}, expected <= 10 (per docstring)"
```

**Failing input**: `n = 1125894277343089729`

## Reproducing the Bug

```python
import dask.utils

n = 1125894277343089729
result = dask.utils.format_bytes(n)

print(f"n = {n}")
print(f"n < 2**60: {n < 2**60}")
print(f"format_bytes({n}) = '{result}'")
print(f"Length: {len(result)}")

assert n < 2**60
assert len(result) > 10
```

Output:
```
n = 1125894277343089729
n < 2**60: True
format_bytes(1125894277343089729) = '1000.00 PiB'
Length: 11
```

## Why This Is A Bug

The docstring explicitly states: "For all values < 2**60, the output is always <= 10 characters." However, for values >= 1000 * 2**50 (approximately 1.126e18), which are still less than 2**60, the function returns strings like "1000.00 PiB" (11 characters), "1001.00 PiB" (11 characters), etc.

This violates the documented contract and could break code that relies on the 10-character guarantee for buffer sizing or formatting.

## Fix

The docstring claim is incorrect. The fix is to either:

1. Update the docstring to reflect the actual behavior (simpler fix):

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -10,7 +10,7 @@ def format_bytes(n: int) -> str:
     >>> format_bytes(1234567890000000)
     '1.10 PiB'

-    For all values < 2**60, the output is always <= 10 characters.
+    For most values, the output is <= 10 characters. Values >= 1000 PiB may exceed this.
     """
```

2. Or modify the implementation to ensure the claim holds (more complex):

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -19,7 +19,11 @@ def format_bytes(n: int) -> str:
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

The first option (documentation fix) is recommended as it's simpler and doesn't change behavior that existing code may depend on.