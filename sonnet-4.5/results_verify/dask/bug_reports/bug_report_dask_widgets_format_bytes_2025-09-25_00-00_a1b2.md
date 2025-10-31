# Bug Report: dask.widgets format_bytes Output Length Violation

**Target**: `dask.widgets.widgets.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function violates its documented guarantee that "For all values < 2**60, the output is always <= 10 characters" when formatting values >= 1000 PiB.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.widgets.widgets import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
def test_format_bytes_length_under_2_60(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = {result!r} has length {len(result)} > 10"
```

**Failing input**: `n = 1_125_894_277_343_089_729` (approximately 1000 PiB)

## Reproducing the Bug

```python
from dask.widgets.widgets import format_bytes

n = 1_125_894_277_343_089_729
result = format_bytes(n)

print(f"n = {n}")
print(f"n < 2**60? {n < 2**60}")
print(f"format_bytes({n}) = {result!r}")
print(f"Length: {len(result)}")

assert n < 2**60
assert len(result) == 11
```

## Why This Is A Bug

The function's docstring explicitly states:

> For all values < 2**60, the output is always <= 10 characters.

However, when n >= 1000 PiB (which is still < 2**60), the output format produces strings like `"1000.00 PiB"` which has 11 characters. This happens because the `.2f` format specifier can produce 4-digit integers (1000-1023) before the decimal point.

## Fix

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -45,7 +45,7 @@ def format_bytes(n: int) -> str:
     >>> format_bytes(1234567890000000)
     '1.10 PiB'

-    For all values < 2**60, the output is always <= 10 characters.
+    For most values < 2**60, the output is typically <= 10 characters (11 for values >= 1000 PiB).
     """
     for prefix, k in (
         ("Pi", 2**50),
```

Alternatively, the formatting could be adjusted to ensure 10 characters max, but this would require more complex logic to handle edge cases.