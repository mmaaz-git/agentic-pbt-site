# Bug Report: Cython.Compiler.Options.normalise_encoding_name Crashes on Null Characters

**Target**: `Cython.Compiler.Options.normalise_encoding_name`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `normalise_encoding_name` function crashes with `ValueError: embedded null character` when given a string containing null bytes, instead of handling it gracefully as it does for other invalid encoding names.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Compiler.Options import normalise_encoding_name


@given(st.text())
def test_normalise_encoding_name_handles_arbitrary_strings(encoding):
    result = normalise_encoding_name('c_string_encoding', encoding)
    assert isinstance(result, str)
```

**Failing input**: `'\x00'`

## Reproducing the Bug

```python
from Cython.Compiler.Options import normalise_encoding_name

result = normalise_encoding_name('c_string_encoding', '\x00')
```

**Output:**
```
ValueError: embedded null character
```

## Why This Is A Bug

The function's documented behavior (see line 323-324 in Options.py) shows that unknown encoding names should be returned as-is:

```python
>>> normalise_encoding_name('c_string_encoding', 'SeriousLyNoSuch--Encoding')
'SeriousLyNoSuch--Encoding'
```

The function catches `LookupError` from `codecs.getdecoder()` to handle unknown encodings (line 335), but it doesn't catch `ValueError`, which is raised when the encoding string contains null characters. This is inconsistent with the documented behavior of gracefully handling invalid encoding names.

## Fix

```diff
--- a/Cython/Compiler/Options.py
+++ b/Cython/Compiler/Options.py
@@ -331,7 +331,7 @@ def normalise_encoding_name(option_name, encoding):

     import codecs
     try:
         decoder = codecs.getdecoder(encoding)
-    except LookupError:
+    except (LookupError, ValueError):
         return encoding  # may exists at runtime ...
     for name in ('ascii', 'utf8'):
         if codecs.getdecoder(name) == decoder:
```