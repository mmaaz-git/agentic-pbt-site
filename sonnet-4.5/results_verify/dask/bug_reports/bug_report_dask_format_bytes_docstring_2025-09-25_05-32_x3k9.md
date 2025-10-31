# Bug Report: dask.utils.format_bytes Docstring Incorrect Character Limit

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The docstring for `format_bytes` claims "For all values < 2**60, the output is always <= 10 characters", but this is violated for values >= 1000 * 2**50 (1000 PiB).

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60))
def test_format_bytes_length_bound(n):
    result = format_bytes(n)
    assert len(result) <= 10
```

**Failing input**: `n=1_125_894_277_343_089_729`

## Reproducing the Bug

```python
from dask.utils import format_bytes

n = 1_125_894_277_343_089_729
result = format_bytes(n)

print(f"Input: {n}")
print(f"Input < 2**60: {n < 2**60}")
print(f"Result: {result!r}")
print(f"Length: {len(result)}")

assert n < 2**60
assert len(result) <= 10
```

Output:
```
Input: 1125894277343089729
Input < 2**60: True
Result: '1000.00 PiB'
Length: 11
AssertionError: 11 <= 10
```

## Why This Is A Bug

The docstring at line 1788 of `dask/utils.py` explicitly states:

> For all values < 2**60, the output is always <= 10 characters.

However, for values in the range [1000 * 2**50, 2**60), the output format is `"XXXX.XX PiB"` which is 11 characters long. For example:
- `format_bytes(999 * 2**50)` → `'999.00 PiB'` (10 chars) ✓
- `format_bytes(1000 * 2**50)` → `'1000.00 PiB'` (11 chars) ✗
- `format_bytes(1024 * 2**50 - 1)` → `'1024.00 PiB'` (11 chars) ✗

This violates the documented API contract.

## Fix

The simplest fix is to update the docstring to reflect the actual behavior:

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1785,7 +1785,7 @@ def format_bytes(n: int) -> str:
     >>> format_bytes(1234567890000000)
     '1.10 PiB'

-    For all values < 2**60, the output is always <= 10 characters.
+    For all values < 2**60, the output is always <= 11 characters.
     """
     for prefix, k in (
         ("Pi", 2**50),
```

Alternatively, the implementation could be modified to ensure 10-character output by using fewer decimal places for large values, but this would change the output format and potentially break existing code that depends on the current format.