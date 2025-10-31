# Bug Report: dask.utils.format_bytes Output Length Exceeds Documented Limit

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function violates its documented guarantee that "For all values < 2**60, the output is always <= 10 characters." Values >= 1000 PiB (but still < 2**60) produce 11-character outputs like "1000.00 PiB".

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.utils import format_bytes

@settings(max_examples=1000)
@given(st.integers(min_value=0, max_value=2**60 - 1))
def test_format_bytes_max_length_10(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)}, expected <= 10"
```

**Failing input**: `n=1125894277343089729` (approximately 1000 PiB)

## Reproducing the Bug

```python
from dask.utils import format_bytes

n = 1125894277343089729
result = format_bytes(n)

print(f"Input: {n} (< 2**60: {n < 2**60})")
print(f"Output: '{result}'")
print(f"Length: {len(result)} characters")
print(f"Expected: <= 10 characters")
print(f"Bug: {len(result)} > 10")
```

Output:
```
Input: 1125894277343089729 (< 2**60: True)
Output: '1000.00 PiB'
Length: 11 characters
Expected: <= 10 characters
Bug: 11 > 10
```

## Why This Is A Bug

The function's docstring explicitly states: "For all values < 2**60, the output is always <= 10 characters." However, any value >= 1000 * 2**50 (1000 PiB = 1,125,899,906,842,624 bytes) produces an 11-character string. Since 1000 PiB < 2**60, this violates the documented contract.

## Fix

The issue occurs because the format string `f"{n / k:.2f} {prefix}B"` produces 4 digits + decimal point + 2 decimals + space + 3-char prefix + "B" = 11 characters when the value is >= 1000 in the given unit.

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1785,7 +1785,10 @@ def format_bytes(n: int) -> str:
         ("ki", 2**10),
     ):
         if n >= k * 0.9:
-            return f"{n / k:.2f} {prefix}B"
+            value = n / k
+            if value >= 1000:
+                return f"{value:.1f} {prefix}B"
+            return f"{value:.2f} {prefix}B"
     return f"{n} B"
```

This fix reduces precision to 1 decimal place for values >= 1000 in their unit, ensuring the output never exceeds 10 characters for values < 2**60.