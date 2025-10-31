# Bug Report: pandas.io.clipboard Encoding Validation Too Strict

**Target**: `pandas.io.clipboard` (specifically `read_clipboard` and `to_clipboard`)
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The encoding validation in `read_clipboard()` and `to_clipboard()` rejects valid UTF-8 encoding names that Python itself accepts, such as "utf_8".

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from pandas.io.clipboards import read_clipboard, to_clipboard

@given(st.sampled_from(["utf-8", "UTF-8", "utf_8", "UTF_8", "Utf-8"]))
def test_clipboard_accepts_python_valid_utf8_encodings(encoding):
    text = "test"
    text.encode(encoding)

    try:
        read_clipboard(encoding=encoding)
    except (NotImplementedError, ValueError) as e:
        if "only supports utf-8 encoding" in str(e):
            raise AssertionError(f"Rejected valid encoding {encoding}: {e}")
```

**Failing input**: `"utf_8"` (and other valid Python UTF-8 encoding names)

## Reproducing the Bug

```python
import pandas as pd
from pandas.io.clipboards import read_clipboard, to_clipboard

text = "hello"
text.encode("utf_8")

try:
    read_clipboard(encoding="utf_8")
except NotImplementedError as e:
    print(f"Bug: {e}")

df = pd.DataFrame([[1, 2], [3, 4]])
try:
    to_clipboard(df, encoding="utf_8")
except ValueError as e:
    print(f"Bug: {e}")
```

## Why This Is A Bug

Python's codec system normalizes encoding names, so "utf-8", "UTF-8", "utf_8", and "UTF_8" are all valid and equivalent ways to specify UTF-8 encoding. The validation logic at `pandas/io/clipboards.py:78-79` and `pandas/io/clipboards.py:160-161` uses:

```python
encoding.lower().replace("-", "") != "utf8"
```

This only removes hyphens but not underscores, so "utf_8" becomes "utf_8" (not "utf8") and gets rejected. This is inconsistent with Python's own encoding system and the error message "only supports utf-8 encoding" which implies all valid UTF-8 encoding names should work.

## Fix

```diff
--- a/pandas/io/clipboards.py
+++ b/pandas/io/clipboards.py
@@ -75,7 +75,7 @@ def read_clipboard(

     # only utf-8 is valid for passed value because that's what clipboard
     # supports
-    if encoding is not None and encoding.lower().replace("-", "") != "utf8":
+    if encoding is not None and encoding.lower().replace("-", "").replace("_", "") != "utf8":
         raise NotImplementedError("reading from clipboard only supports utf-8 encoding")

     check_dtype_backend(dtype_backend)
@@ -157,7 +157,7 @@ def to_clipboard(
     encoding = kwargs.pop("encoding", "utf-8")

     # testing if an invalid encoding is passed to clipboard
-    if encoding is not None and encoding.lower().replace("-", "") != "utf8":
+    if encoding is not None and encoding.lower().replace("-", "").replace("_", "") != "utf8":
         raise ValueError("clipboard only supports utf-8 encoding")

     from pandas.io.clipboard import clipboard_set
```