# Bug Report: dask.utils.parse_bytes Accepts Negative Values

**Target**: `dask.utils.parse_bytes`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `parse_bytes` function accepts negative numeric values and returns negative byte sizes, which is semantically incorrect since byte sizes cannot be negative.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.utils import parse_bytes


@given(
    st.integers(max_value=-1),
    st.sampled_from(['kB', 'MB', 'GB', 'KiB', 'MiB', 'GiB', 'B', ''])
)
def test_parse_bytes_rejects_negative_strings(n, unit):
    s = f"{n}{unit}"
    result = parse_bytes(s)
    assert result >= 0, f"parse_bytes('{s}') returned negative value {result}"
```

**Failing input**: `n=-1, unit='kB'`

## Reproducing the Bug

```python
from dask.utils import parse_bytes

print(parse_bytes("-128MiB"))
print(parse_bytes(-100))
print(parse_bytes("-5kB"))
```

Output:
```
-134217728
-100
-5000
```

## Why This Is A Bug

Byte sizes represent amounts of data and are inherently non-negative. The function's name and purpose (parsing byte size strings) implies it should return non-negative values. Allowing negative values can cause downstream issues:

1. In `dask.bytes.read_bytes`, `parse_bytes` is used to parse `blocksize` and `sample` parameters (lines 90 and 167 of core.py)
2. A negative `blocksize` would cause incorrect behavior in the block calculation logic (lines 124-143)
3. The docstring provides no examples of negative values, suggesting they're not intended

## Fix

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1612,6 +1612,8 @@ def parse_bytes(s: float | str) -> int:
         ValueError: Could not interpret 'foos' as a byte unit
     """
     if isinstance(s, (int, float)):
+        if s < 0:
+            raise ValueError("Byte size cannot be negative")
         return int(s)
     s = s.replace(" ", "")
     if not any(char.isdigit() for char in s):
@@ -1636,6 +1638,8 @@ def parse_bytes(s: float | str) -> int:
         raise ValueError("Could not interpret '%s' as a byte unit" % suffix) from e

     result = n * multiplier
+    if result < 0:
+        raise ValueError("Byte size cannot be negative")
     return int(result)
```