# Bug Report: dask.utils.format_bytes violates documented length guarantee

**Target**: `dask.utils.format_bytes`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function's docstring claims "For all values < 2**60, the output is always <= 10 characters", but this claim is violated for values >= 1000 PiB (approximately 1.126e18 bytes).

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=1000)
def test_format_bytes_length_claim(n):
    """
    Test the documented claim: "For all values < 2**60, the output is always <= 10 characters."
    """
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)}, expected <= 10"
```

**Failing input**: `n=1_125_894_277_343_089_729` (approximately 1000 PiB)

## Reproducing the Bug

```python
from dask.utils import format_bytes

result = format_bytes(1_125_899_906_842_624_000)
print(f"Result: '{result}'")
print(f"Length: {len(result)} characters")

assert len(result) <= 10, f"Expected <= 10 characters, got {len(result)}"
```

Output:
```
Result: '1000.00 PiB'
Length: 11 characters
AssertionError: Expected <= 10 characters, got 11
```

Additional examples:
- `format_bytes(999 * 2**50)` = `'999.00 PiB'` (10 chars) ✓
- `format_bytes(1000 * 2**50)` = `'1000.00 PiB'` (11 chars) ✗
- `format_bytes(1023 * 2**50)` = `'1023.00 PiB'` (11 chars) ✗

## Why This Is A Bug

The function's docstring explicitly states:

> For all values < 2**60, the output is always <= 10 characters.

However, for values >= 1000 PiB (which are still < 2**60), the output exceeds 10 characters. This violates the documented API contract that users may rely on for formatting constraints (e.g., fixed-width displays).

The root cause is that the formatting uses `.2f` which always produces 2 decimal places. For values >= 1000, the formatted number is 7 characters (e.g., "1000.00"), and adding " PiB" (4 characters) gives 11 total.

## Fix

The fix requires reducing precision for larger values to stay within the 10-character limit:

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -123,7 +123,11 @@ def format_bytes(n: int) -> str:
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

With this fix:
- `format_bytes(999 * 2**50)` = `'999.00 PiB'` (10 chars)
- `format_bytes(1000 * 2**50)` = `'1000.0 PiB'` (10 chars)
- `format_bytes(1023 * 2**50)` = `'1023.0 PiB'` (10 chars)