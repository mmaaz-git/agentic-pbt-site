# Bug Report: pandas.io.clipboard Invalid Regex Separator Crash

**Target**: `pandas.io.clipboard.read_clipboard` and `pandas.read_csv`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`read_clipboard()` (and `read_csv()`) crash with confusing regex errors when multi-character separators contain regex metacharacters. Common separators like `**`, `++`, or `0(` cause `re.PatternError` exceptions instead of being treated as literal strings or providing helpful error messages.

## Property-Based Test

```python
from hypothesis import given, assume, strategies as st
from unittest.mock import patch
import pandas as pd
import pytest

@given(st.text(min_size=2, max_size=5))
def test_multi_char_sep_uses_python_engine(sep):
    """Multi-character separators should either work or give helpful errors"""
    from pandas.io.clipboards import read_clipboard

    assume(len(sep) > 1)

    test_data = f"a{sep}b\n1{sep}2\n3{sep}4"

    with patch('pandas.io.clipboard.clipboard_get', return_value=test_data):
        try:
            result = read_clipboard(sep=sep)
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            # Should not crash with low-level regex errors
            assert not isinstance(e, re.PatternError), \
                f"Separator {repr(sep)} caused regex error instead of helpful message"
```

**Failing inputs**: `'0('`, `'0)'`, `'**'`, `'++'`, `'[['`, and many others containing regex metacharacters

## Reproducing the Bug

```python
from unittest.mock import patch
import pandas as pd

test_data = "a**b\n1**2\n3**4"

with patch('pandas.io.clipboard.clipboard_get', return_value=test_data):
    result = pd.read_clipboard(sep='**')
```

**Output**:
```
re.PatternError: nothing to repeat at position 0
```

**Similar failure with other separators**:
```python
pd.read_clipboard(sep='0(')  # PatternError: missing ), unterminated subpattern
pd.read_clipboard(sep='++')  # PatternError: nothing to repeat at position 1
pd.read_clipboard(sep='[[')  # PatternError: unterminated character set
```

## Why This Is A Bug

1. **Poor UX**: Users expect multi-character separators to work as literal strings (like `**` or `::`)
2. **Confusing errors**: Low-level `re.PatternError` messages don't explain what went wrong
3. **Inconsistent behavior**: Some multi-char seps work (`::`, `||`), others crash (`**`, `++`)
4. **No documentation**: The docstring doesn't explain that separators containing regex metacharacters must be escaped
5. **No workaround guidance**: Error message doesn't suggest using `re.escape()` or specifying a valid regex

The documentation states "string or regex delimiter" but doesn't clarify:
- When a string is treated as regex vs literal
- That multi-character strings are always treated as regex
- How to use literal multi-character separators containing metacharacters

## Fix

**Option 1: Better error handling** (Recommended)

```diff
diff --git a/pandas/io/clipboards.py b/pandas/io/clipboards.py
index abcd1234..efgh5678 100644
--- a/pandas/io/clipboards.py
+++ b/pandas/io/clipboards.py
@@ -115,6 +115,15 @@ def read_clipboard(
     # Regex separator currently only works with python engine.
     # Default to python if separator is multi-character (regex)
     if len(sep) > 1 and kwargs.get("engine") is None:
+        # Validate that the separator is a valid regex pattern
+        import re
+        try:
+            re.compile(sep)
+        except re.error as e:
+            raise ValueError(
+                f"Multi-character separator {sep!r} is not a valid regular expression. "
+                f"If you want to use it as a literal string, escape it with: re.escape({sep!r})"
+            ) from e
         kwargs["engine"] = "python"
     elif len(sep) > 1 and kwargs.get("engine") == "c":
         warnings.warn(
```

**Option 2: Auto-escape literal separators**

Automatically escape separators that aren't already valid regex patterns:

```diff
diff --git a/pandas/io/clipboards.py b/pandas/io/clipboards.py
index abcd1234..efgh5678 100644
--- a/pandas/io/clipboards.py
+++ b/pandas/io/clipboards.py
@@ -115,6 +115,14 @@ def read_clipboard(
     # Regex separator currently only works with python engine.
     # Default to python if separator is multi-character (regex)
     if len(sep) > 1 and kwargs.get("engine") is None:
+        # Try to compile as regex; if it fails, escape it
+        import re
+        try:
+            re.compile(sep)
+        except re.error:
+            # Not a valid regex, treat as literal string
+            sep = re.escape(sep)
+            warnings.warn(f"Separator treated as literal string, not regex", stacklevel=2)
         kwargs["engine"] = "python"
```

**Option 3: Documentation improvement**

Update the docstring to explicitly explain the regex behavior:

```diff
-sep : str, default '\\s+'
-    A string or regex delimiter. The default of ``'\\s+'`` denotes
-    one or more whitespace characters.
+sep : str, default '\\s+'
+    A string or regex delimiter. The default of ``'\\s+'`` denotes
+    one or more whitespace characters.
+
+    Note: Multi-character separators (len > 1) are always interpreted as
+    regular expression patterns. If your separator contains regex
+    metacharacters (like `*`, `+`, `(`, etc.), escape it using
+    ``import re; sep = re.escape(your_separator)``.
```

All three options could be combined for the best user experience.