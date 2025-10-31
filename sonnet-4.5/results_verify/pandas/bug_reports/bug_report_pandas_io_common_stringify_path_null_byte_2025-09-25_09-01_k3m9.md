# Bug Report: pandas.io.common Null Byte Crash in File Path Handling

**Target**: `pandas.io.common.stringify_path` / `pandas.io.common.infer_compression`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

File path functions crash with `ValueError: embedded null byte` when given paths containing null bytes, instead of handling them gracefully or providing a more informative error message.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.io.common as io_common

@given(st.text())
def test_infer_compression_deterministic(filepath):
    result1 = io_common.infer_compression(filepath, compression='infer')
    result2 = io_common.infer_compression(filepath, compression='infer')
    assert result1 == result2
```

**Failing input**: `filepath='~\x00'`

## Reproducing the Bug

```python
import pandas.io.common as io_common
import pandas as pd

filepath = '~\x00'

try:
    result = io_common.stringify_path(filepath)
except ValueError as e:
    print(f'stringify_path crashed: {e}')

try:
    result = io_common.infer_compression(filepath, compression='infer')
except ValueError as e:
    print(f'infer_compression crashed: {e}')

try:
    df = pd.read_csv(filepath)
except ValueError as e:
    print(f'read_csv crashed: {e}')
```

Output:
```
stringify_path crashed: embedded null byte
infer_compression crashed: embedded null byte
read_csv crashed: embedded null byte
```

## Why This Is A Bug

While null bytes in file paths are invalid on most filesystems, the function should either:
1. Validate input and raise a clear, informative error (e.g., `ValueError: Invalid file path: contains null byte`)
2. Sanitize the input before passing it to `os.path.expanduser`

Instead, it crashes deep in the call stack with a cryptic error from `os.path.expanduser`, making it harder for users to understand what went wrong.

This affects all pandas I/O functions that use `stringify_path`, including `read_csv`, `read_json`, `read_excel`, etc.

## Fix

```diff
diff --git a/pandas/io/common.py b/pandas/io/common.py
index 1234567..abcdefg 100644
--- a/pandas/io/common.py
+++ b/pandas/io/common.py
@@ -200,6 +200,9 @@ def _expand_user(filepath_or_buffer: FilePath | BaseBuffer) -> FilePath | BaseB
     if not isinstance(filepath_or_buffer, str):
         return filepath_or_buffer

+    if '\x00' in filepath_or_buffer:
+        raise ValueError(f"Invalid file path: contains null byte")
+
     # need to normalize the pathname, but we need to use
     # the original path to detect the protocol
     return os.path.expanduser(filepath_or_buffer)
```

This provides a clearer error message and catches the problem earlier in the call stack.