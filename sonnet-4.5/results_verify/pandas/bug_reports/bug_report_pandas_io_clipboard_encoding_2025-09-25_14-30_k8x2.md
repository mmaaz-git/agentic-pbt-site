# Bug Report: pandas.io.clipboard Encoding Validation Inconsistency

**Target**: `pandas.io.clipboard` (`read_clipboard` and `to_clipboard` functions)
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `read_clipboard` and `to_clipboard` functions perform identical encoding validation but raise different exception types: `read_clipboard` raises `NotImplementedError` while `to_clipboard` raises `ValueError` for non-UTF-8 encodings. This inconsistency violates the principle of least surprise and makes error handling unnecessarily complex for users.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from pandas import DataFrame
from pandas.io.clipboards import read_clipboard, to_clipboard
from unittest.mock import patch


@given(encoding=st.sampled_from(['ascii', 'latin-1', 'iso-8859-1', 'cp1252']))
def test_encoding_validation_consistency(encoding):
    df = DataFrame({'a': [1, 2, 3]})

    read_exception = None
    to_exception = None

    with patch('pandas.io.clipboards.clipboard_get', return_value='a\n1\n2\n3'):
        try:
            read_clipboard(encoding=encoding)
        except Exception as e:
            read_exception = type(e)

    try:
        to_clipboard(df, encoding=encoding)
    except Exception as e:
        to_exception = type(e)

    assert read_exception == to_exception, (
        f"Inconsistent exceptions: read_clipboard raises {read_exception.__name__}, "
        f"but to_clipboard raises {to_exception.__name__}"
    )
```

**Failing input**: `encoding='ascii'` (and any non-UTF-8 encoding)

## Reproducing the Bug

```python
import pandas as pd
from pandas import DataFrame
from unittest.mock import patch

df = DataFrame({'a': [1, 2, 3]})

with patch('pandas.io.clipboards.clipboard_get', return_value='a\n1\n2\n3'):
    try:
        pd.read_clipboard(encoding='ascii')
    except Exception as e:
        print(f"read_clipboard: {type(e).__name__}")

try:
    df.to_clipboard(encoding='ascii')
except Exception as e:
    print(f"to_clipboard: {type(e).__name__}")
```

Output:
```
read_clipboard: NotImplementedError
to_clipboard: ValueError
```

## Why This Is A Bug

Both functions perform the exact same validation check (`encoding.lower().replace("-", "") != "utf8"`), but they raise different exception types. This creates an inconsistent API where users must catch different exceptions for the same conceptual error. The functions should raise the same exception type for the same validation failure.

From a design perspective, `NotImplementedError` is more appropriate here since it signals that non-UTF-8 encodings are not supported (not that the input is invalid), but consistency is more important than the specific choice.

## Fix

```diff
--- a/pandas/io/clipboards.py
+++ b/pandas/io/clipboards.py
@@ -157,7 +157,7 @@ def to_clipboard(
     encoding = kwargs.pop("encoding", "utf-8")

     # testing if an invalid encoding is passed to clipboard
     if encoding is not None and encoding.lower().replace("-", "") != "utf8":
-        raise ValueError("clipboard only supports utf-8 encoding")
+        raise NotImplementedError("clipboard only supports utf-8 encoding")

     from pandas.io.clipboard import clipboard_set
```

Alternatively, both could be changed to `ValueError` if that's preferred, but `NotImplementedError` better conveys that the feature is not implemented rather than the input being invalid.