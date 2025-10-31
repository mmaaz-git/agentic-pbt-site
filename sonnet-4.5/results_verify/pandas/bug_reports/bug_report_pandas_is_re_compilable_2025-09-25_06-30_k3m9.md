# Bug Report: pandas.api.types.is_re_compilable Raises Exception Instead of Returning False

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_re_compilable()` function is documented to return a boolean indicating whether an object can be compiled as a regex pattern. However, it raises `re.PatternError` for invalid regex patterns instead of returning `False`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.api.types as pat

@given(st.text())
def test_is_re_compilable(text):
    result = pat.is_re_compilable(text)
    if result:
        import re
        try:
            re.compile(text)
        except Exception as e:
            raise AssertionError(f"is_re_compilable returned True but re.compile failed: {e}")
```

**Failing input**: `'['` (and other invalid regex patterns like `'('`, `')'`, `'?'`, `'*'`, `'+'`)

## Reproducing the Bug

```python
import pandas.api.types as pat

print(pat.is_re_compilable('['))
```

**Expected output**: `False`

**Actual output**:
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/path/to/pandas/core/dtypes/inference.py", line 188, in is_re_compilable
    re.compile(obj)
re.PatternError: unterminated character set at position 0
```

## Why This Is A Bug

The function's docstring states it should "Check if the object can be compiled into a regex pattern instance" and return a bool. The function name itself (`is_re_compilable`) implies it performs a boolean check. Invalid regex patterns should return `False`, not raise exceptions.

The current implementation only catches `TypeError`, but `re.compile()` can also raise `re.PatternError` for syntactically invalid regex patterns.

## Fix

```diff
--- a/pandas/core/dtypes/inference.py
+++ b/pandas/core/dtypes/inference.py
@@ -185,7 +185,7 @@ def is_re_compilable(obj) -> bool:
     False
     """
     try:
         re.compile(obj)
-    except TypeError:
+    except (TypeError, re.error):
         return False
     else:
         return True
```