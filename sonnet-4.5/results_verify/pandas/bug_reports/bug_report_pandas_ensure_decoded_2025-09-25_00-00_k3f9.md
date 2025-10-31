# Bug Report: pandas.core.computation.common.ensure_decoded - Invalid UTF-8 bytes cause UnicodeDecodeError

**Target**: `pandas.core.computation.common.ensure_decoded`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `ensure_decoded` function crashes with `UnicodeDecodeError` when given bytes that are not valid UTF-8, instead of handling the error gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.computation.common import ensure_decoded

@given(st.binary(min_size=0, max_size=100))
def test_ensure_decoded_bytes_to_str(b):
    result = ensure_decoded(b)
    assert isinstance(result, str)
```

**Failing input**: `b'\x80'`

## Reproducing the Bug

```python
from pandas.core.computation.common import ensure_decoded

invalid_utf8 = b'\x80'
result = ensure_decoded(invalid_utf8)
```

## Why This Is A Bug

The function is named `ensure_decoded`, which implies it should robustly handle byte-to-string conversion. The docstring states "If we have bytes, decode them to unicode" without documenting any preconditions about the bytes being valid UTF-8.

While in practice the function is called on PyTables metadata that should be valid UTF-8, corrupt HDF5 files or edge cases could trigger this crash. The function should handle invalid bytes gracefully rather than crashing.

## Fix

```diff
--- a/pandas/core/computation/common.py
+++ b/pandas/core/computation/common.py
@@ -12,7 +12,7 @@ def ensure_decoded(s) -> str:
     If we have bytes, decode them to unicode.
     """
     if isinstance(s, (np.bytes_, bytes)):
-        s = s.decode(get_option("display.encoding"))
+        s = s.decode(get_option("display.encoding"), errors='replace')
     return s
```