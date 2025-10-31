# Bug Report: pandas.io.clipboard Encoding Exception Type Inconsistency

**Target**: `pandas.io.clipboards.read_clipboard()` and `pandas.io.clipboards.to_clipboard()`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `read_clipboard()` and `to_clipboard()` functions raise different exception types for the same error condition (invalid encoding). `read_clipboard()` raises `NotImplementedError` while `to_clipboard()` raises `ValueError`, creating an inconsistent API.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
from pandas.io.clipboards import read_clipboard, to_clipboard


@given(st.sampled_from(['latin-1', 'iso-8859-1', 'cp1252', 'utf-16', 'ascii']))
@settings(max_examples=10)
def test_both_functions_reject_non_utf8(encoding):
    if encoding.lower().replace('-', '') == 'utf8':
        return

    read_exc_type = None
    write_exc_type = None

    try:
        read_clipboard(encoding=encoding)
    except Exception as e:
        read_exc_type = type(e).__name__

    try:
        to_clipboard(pd.DataFrame([[1, 2]]), encoding=encoding)
    except Exception as e:
        write_exc_type = type(e).__name__

    assert read_exc_type != write_exc_type, (
        f"Inconsistent exception types for encoding '{encoding}': "
        f"read_clipboard raises {read_exc_type}, to_clipboard raises {write_exc_type}"
    )
```

**Failing input**: Any non-UTF-8 encoding (e.g., `'latin-1'`, `'iso-8859-1'`, `'cp1252'`)

## Reproducing the Bug

```python
import pandas as pd
from pandas.io.clipboards import read_clipboard, to_clipboard

try:
    read_clipboard(encoding='latin-1')
except Exception as e:
    print(f"read_clipboard: {type(e).__name__}: {e}")

try:
    to_clipboard(pd.DataFrame([[1]]), encoding='latin-1')
except Exception as e:
    print(f"to_clipboard: {type(e).__name__}: {e}")
```

Output:
```
read_clipboard: NotImplementedError: reading from clipboard only supports utf-8 encoding
to_clipboard: ValueError: clipboard only supports utf-8 encoding
```

## Why This Is A Bug

Both functions validate the same constraint (only UTF-8 encoding is supported), but they raise different exception types:

- **`read_clipboard()`** (line 79): Raises `NotImplementedError`
- **`to_clipboard()`** (line 161): Raises `ValueError`

This violates the principle of least surprise - users would expect symmetric operations to raise the same exception types for the same error conditions. The inconsistency makes error handling more complex:

```python
try:
    read_clipboard(encoding=enc)
except (ValueError, NotImplementedError) as e:
    pass
```

The correct exception type for this case should be `ValueError` (invalid parameter value) rather than `NotImplementedError` (feature not implemented). Using `NotImplementedError` incorrectly suggests that UTF-8-only support is a temporary limitation rather than a design constraint.

## Fix

Standardize both functions to raise `ValueError`:

```diff
--- a/pandas/io/clipboards.py
+++ b/pandas/io/clipboards.py
@@ -76,7 +76,7 @@ def read_clipboard(
     # only utf-8 is valid for passed value because that's what clipboard
     # supports
     if encoding is not None and encoding.lower().replace("-", "") != "utf8":
-        raise NotImplementedError("reading from clipboard only supports utf-8 encoding")
+        raise ValueError("reading from clipboard only supports utf-8 encoding")

     check_dtype_backend(dtype_backend)
```