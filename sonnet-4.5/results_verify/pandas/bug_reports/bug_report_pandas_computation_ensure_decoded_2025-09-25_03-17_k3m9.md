# Bug Report: pandas.core.computation.common.ensure_decoded UnicodeDecodeError

**Target**: `pandas.core.computation.common.ensure_decoded`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `ensure_decoded` function crashes with `UnicodeDecodeError` when given bytes containing invalid UTF-8 sequences, instead of handling the error gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.computation.common import ensure_decoded

@given(st.binary())
def test_ensure_decoded_returns_str(data):
    result = ensure_decoded(data)
    assert isinstance(result, str)
```

**Failing input**: `b'\x80'`

## Reproducing the Bug

```python
from pandas.core.computation.common import ensure_decoded

data = b'\x80'
result = ensure_decoded(data)
```

Output:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
```

## Why This Is A Bug

The `ensure_decoded` function is used to decode bytes from HDF5/PyTables files. While pandas-written files should always contain valid UTF-8, manually created or corrupted HDF5 files could contain invalid byte sequences. The function should handle this gracefully with an error handler (e.g., 'replace', 'ignore', or 'backslashreplace') rather than crashing.

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