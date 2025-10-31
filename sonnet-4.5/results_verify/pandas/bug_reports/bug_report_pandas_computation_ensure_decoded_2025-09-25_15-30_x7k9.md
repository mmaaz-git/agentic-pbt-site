# Bug Report: pandas.core.computation.common.ensure_decoded UnicodeDecodeError

**Target**: `pandas.core.computation.common.ensure_decoded`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `ensure_decoded` function crashes with `UnicodeDecodeError` when given bytes that are not valid UTF-8, violating its contract to always return a `str`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.computation.common import ensure_decoded

@given(st.binary(min_size=1))
def test_ensure_decoded_bytes(b):
    result = ensure_decoded(b)
    assert isinstance(result, str)
```

**Failing input**: `b'\x80'`

## Reproducing the Bug

```python
from pandas.core.computation.common import ensure_decoded

invalid_utf8_bytes = b'\x80'
result = ensure_decoded(invalid_utf8_bytes)
```

Output:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```

## Why This Is A Bug

The function is annotated to return `str` and is used in contexts where it must handle arbitrary bytes (e.g., from PyTables HDF5 metadata). When it encounters bytes that are not valid UTF-8, it crashes instead of gracefully handling the decoding error. The function's callers in `pytables.py` rely on it to decode metadata that could come from external sources and may not be valid UTF-8.

## Fix

```diff
--- a/pandas/core/computation/common.py
+++ b/pandas/core/computation/common.py
@@ -12,5 +12,5 @@ def ensure_decoded(s) -> str:
     If we have bytes, decode them to unicode.
     """
     if isinstance(s, (np.bytes_, bytes)):
-        s = s.decode(get_option("display.encoding"))
+        s = s.decode(get_option("display.encoding"), errors='replace')
     return s
```