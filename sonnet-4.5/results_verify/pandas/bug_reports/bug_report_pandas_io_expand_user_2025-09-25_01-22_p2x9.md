# Bug Report: pandas.io.common._expand_user Crashes on Null Bytes

**Target**: `pandas.io.common._expand_user`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_expand_user` function crashes with `ValueError: embedded null byte` when given strings containing null bytes, instead of returning the input unchanged as documented.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.io.common as common


@given(filepath=st.text(min_size=1))
def test_expand_user_idempotence(filepath):
    result1 = common._expand_user(filepath)
    result2 = common._expand_user(result1)
    assert result1 == result2
```

**Failing input**: `filepath='~\x00'`

## Reproducing the Bug

```python
import pandas.io.common as common

filepath = '~\x00'
result = common._expand_user(filepath)
```

**Output:**
```
ValueError: embedded null byte
```

## Why This Is A Bug

The `_expand_user` function is documented to return "an expanded filepath or the input if not expandable." However, it crashes on strings with embedded null bytes instead of handling them gracefully.

While `_expand_user` is an internal function (starts with `_`), it is called by public functions like `stringify_path` and `_get_filepath_or_buffer`, which propagates the crash to user-facing pandas I/O operations. Although null bytes in file paths are unusual, they can appear in malformed user input (e.g., from web forms or untrusted sources), and the function should handle them gracefully rather than crashing.

## Fix

```diff
--- a/pandas/io/common.py
+++ b/pandas/io/common.py
@@ -200,7 +200,10 @@ def _expand_user(filepath_or_buffer: str | BaseBufferT) -> str | BaseBufferT:
                                   input if not expandable
     """
     if isinstance(filepath_or_buffer, str):
-        return os.path.expanduser(filepath_or_buffer)
+        try:
+            return os.path.expanduser(filepath_or_buffer)
+        except ValueError:
+            return filepath_or_buffer
     return filepath_or_buffer
```