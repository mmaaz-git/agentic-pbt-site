# Bug Report: pandas.core.computation.common.ensure_decoded Crashes on Invalid UTF-8

**Target**: `pandas.core.computation.common.ensure_decoded`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `ensure_decoded` function crashes with a `UnicodeDecodeError` when given bytes that are not valid UTF-8, despite claiming to handle bytes by decoding them to unicode.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.computation.common import ensure_decoded

@given(st.one_of(st.binary(), st.text()))
def test_ensure_decoded_idempotence(s):
    result1 = ensure_decoded(s)
    result2 = ensure_decoded(result1)
    assert result1 == result2
```

**Failing input**: `b'\x80'`

## Reproducing the Bug

```python
from pandas.core.computation.common import ensure_decoded

result = ensure_decoded(b'\x80')
```

## Why This Is A Bug

The function's docstring states "If we have bytes, decode them to unicode" but doesn't mention that it only handles valid UTF-8 bytes. The function crashes instead of handling invalid UTF-8 gracefully (e.g., using error handlers like 'replace' or 'ignore').

## Fix

```diff
diff --git a/pandas/core/computation/common.py b/pandas/core/computation/common.py
index abc123..def456 100644
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