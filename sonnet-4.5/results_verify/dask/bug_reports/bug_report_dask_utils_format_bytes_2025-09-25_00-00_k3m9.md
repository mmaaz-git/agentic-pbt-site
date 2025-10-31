# Bug Report: dask.utils.format_bytes Output Length Violation

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function's documentation claims "For all values < 2**60, the output is always <= 10 characters", but this is violated for values >= 1000 PiB (1,125,899,906,842,624,000 bytes), which produce 11-character output like '1000.00 PiB'.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.utils import format_bytes


@given(st.integers(min_value=1, max_value=2**60))
def test_format_bytes_output_length(n):
    result = format_bytes(n)
    assert len(result) <= 10
```

**Failing input**: `n=1_125_894_277_343_089_729`

## Reproducing the Bug

```python
from dask.utils import format_bytes

n = 1_125_899_906_842_624_000
result = format_bytes(n)

print(f"n = {n}")
print(f"n < 2**60 = {n < 2**60}")
print(f"result = {result!r}")
print(f"len(result) = {len(result)}")

assert n < 2**60
assert len(result) == 11
assert result == '1000.00 PiB'
```

Output:
```
n = 1125899906842624000
n < 2**60 = True
result = '1000.00 PiB'
len(result) = 11
```

## Why This Is A Bug

The function's docstring explicitly states: "For all values < 2**60, the output is always <= 10 characters." This is a contract violation - the API documentation makes a promise that the implementation doesn't keep.

The issue occurs because the function formats numbers with 2 decimal places using `f"{n / k:.2f} {prefix}B"`. When `n / k >= 1000` (i.e., when n >= 1000 * 2^50), the output becomes "1000.00 PiB" which is 11 characters. Since 1000 * 2^50 = 1,125,899,906,842,624,000 < 2^60, this violates the documented guarantee.

## Fix

The fix depends on whether the documentation or implementation should change:

**Option 1: Fix the documentation** (simplest)
```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -15,7 +15,7 @@
     >>> format_bytes(1234567890000000)
     '1.10 PiB'

-    For all values < 2**60, the output is always <= 10 characters.
+    For most values < 2**60, the output is typically <= 10 characters.
     """
```

**Option 2: Fix the implementation** (preserves the guarantee)
```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -24,7 +24,10 @@
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

The second option maintains the 10-character guarantee by using 1 decimal place (e.g., "1000.0 PiB") when the value reaches 1000 or above.