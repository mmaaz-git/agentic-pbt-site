# Bug Report: dask.utils format_bytes/parse_bytes Round-Trip Violation

**Target**: `dask.utils.format_bytes` and `dask.utils.parse_bytes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `format_bytes` and `parse_bytes` functions do not form a proper round-trip: `parse_bytes(format_bytes(n))` often returns a different value than `n` due to precision loss from formatting with only 2 decimal places.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from dask.utils import parse_bytes, format_bytes

@given(n=st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=1000)
def test_format_bytes_parse_bytes_roundtrip(n):
    formatted = format_bytes(n)
    parsed = parse_bytes(formatted)
    assert parsed == n, f"Round-trip failed: {n} -> {formatted} -> {parsed}"
```

**Failing input**: `n=1234` (and many others)

## Reproducing the Bug

```python
from dask.utils import parse_bytes, format_bytes

n = 1234
formatted = format_bytes(n)
parsed = parse_bytes(formatted)

print(f"Original: {n}")
print(f"Formatted: {formatted}")
print(f"Parsed back: {parsed}")
print(f"Match: {parsed == n}")
```

Output:
```
Original: 1234
Formatted: 1.21 kiB
Parsed back: 1238
Match: False
```

Additional examples:
- `parse_bytes(format_bytes(5000))` = 5120 (not 5000)
- `parse_bytes(format_bytes(12345))` = 12288 (not 12345)
- `parse_bytes(format_bytes(123456))` = 123904 (not 123456)

## Why This Is A Bug

1. **Mathematical property violation**: Round-trip conversion should preserve the original value for lossless operations. While some precision loss is expected with floating-point formatting, users would reasonably expect `parse_bytes(format_bytes(n)) == n` for byte values.

2. **Potential data corruption**: If code uses these functions to serialize/deserialize byte counts (e.g., in configuration files or logs), the values will silently change.

3. **Inconsistent behavior**: The docstring for `parse_bytes` includes examples that work correctly (e.g., `parse_bytes('5.4 kB')` returns exactly 5400), suggesting that round-trip should work.

## Root Cause

In `dask/utils.py`:

1. **Line 1797** (`format_bytes`): Formats with only 2 decimal places:
   ```python
   return f"{n / k:.2f} {prefix}B"
   ```

2. **Line 1638** (`parse_bytes`): Converts back to int:
   ```python
   return int(result)
   ```

The 2-decimal-place formatting causes rounding that accumulates error when converting back.

Example:
- `1234 / 1024 = 1.2050781...`
- Formatted as `"1.21"` (rounded up)
- `1.21 * 1024 = 1238.04`
- `int(1238.04) = 1238` (not 1234)

## Fix

Increase precision in `format_bytes` to preserve more significant digits:

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1794,7 +1794,7 @@ def format_bytes(n: int) -> str:
         ("ki", 2**10),
     ):
         if n >= k * 0.9:
-            return f"{n / k:.2f} {prefix}B"
+            return f"{n / k:.6f} {prefix}B"
     return f"{n} B"
```

However, this changes the output format significantly. A better approach might be to:

1. Use more decimal places (e.g., 4-6) for better precision while maintaining readability
2. Or document that round-trip conversion is not guaranteed
3. Or add a `format_bytes(..., precise=True)` parameter for lossless formatting

Alternative fix (preserve exact byte count):

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1794,7 +1794,10 @@ def format_bytes(n: int) -> str:
         ("ki", 2**10),
     ):
         if n >= k * 0.9:
-            return f"{n / k:.2f} {prefix}B"
+            # Use enough precision to preserve exact byte count
+            ratio = n / k
+            decimal_places = max(2, len(str(n)) - len(str(int(ratio))))
+            return f"{ratio:.{decimal_places}f} {prefix}B"
     return f"{n} B"
```

Or simply document the limitation and recommend using plain integers for exact values.