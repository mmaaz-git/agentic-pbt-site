# Bug Report: dask.utils.format_bytes Length Exceeds Documented Limit

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`format_bytes` violates its documented guarantee that "For all values < 2**60, the output is always <= 10 characters." Values >= 1000 PiB produce 11-character strings.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import dask.utils

@given(st.integers(min_value=0, max_value=2**60))
@settings(max_examples=500)
def test_format_bytes_length_claim(n):
    formatted = dask.utils.format_bytes(n)
    assert len(formatted) <= 10
```

**Failing input**: `1125894277343089729` (which equals approximately 1000 PiB)

## Reproducing the Bug

```python
import dask.utils

n = 1125899906842624000
result = dask.utils.format_bytes(n)
print(f"format_bytes({n}) = {result!r}")
print(f"Length: {len(result)}")
```

Output:
```
format_bytes(1125899906842624000) = '1000.00 PiB'
Length: 11
```

Additional examples:
```python
format_bytes(1124774006935781376)  # 999 PiB -> '999.00 PiB' (10 chars) ✓
format_bytes(1125899906842624000)  # 1000 PiB -> '1000.00 PiB' (11 chars) ✗
format_bytes(1151795604700004352)  # 1023 PiB -> '1023.00 PiB' (11 chars) ✗
```

## Why This Is A Bug

The docstring explicitly states: "For all values < 2**60, the output is always <= 10 characters."

The value `1125899906842624000` is less than `2**60` (which is `1152921504606846976`), yet the output `'1000.00 PiB'` is 11 characters, violating the documented contract.

## Fix

The function should switch to the next unit (EiB in this case) when the value would produce a 4-digit mantissa, or adjust the formatting to use fewer decimal places for large values:

Option 1: Switch to EiB for values >= 1024 PiB
```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -xxx,x +xxx,x @@ def format_bytes(n: int) -> str:

-    for prefix in ["kiB", "MiB", "GiB", "TiB", "PiB"]:
+    for prefix in ["kiB", "MiB", "GiB", "TiB", "PiB", "EiB"]:
         n /= 1024.0
         if abs(n) < 1024.0:
             return f"{n:.2f} {prefix}"
```

Option 2: Adjust decimal places to maintain <= 10 characters
```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -xxx,x +xxx,x @@ def format_bytes(n: int) -> str:
     for prefix in ["kiB", "MiB", "GiB", "TiB", "PiB"]:
         n /= 1024.0
         if abs(n) < 1024.0:
-            return f"{n:.2f} {prefix}"
+            if abs(n) >= 1000.0:
+                return f"{n:.1f} {prefix}"
+            else:
+                return f"{n:.2f} {prefix}"
```