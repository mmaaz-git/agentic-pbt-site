# Bug Report: pandas.api.types.is_re_compilable Crashes on Invalid Regex Patterns

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_re_compilable()` function crashes with `re.PatternError` on invalid regex patterns like `'\'` and `'['` instead of returning `False`. The function only catches `TypeError` but `re.compile()` can also raise `re.PatternError` for malformed patterns.

## Property-Based Test

```python
import re
from hypothesis import given, strategies as st, settings
from pandas.api.types import is_re_compilable


@given(st.one_of(
    st.text(),
    st.binary(),
    st.integers(),
    st.floats(),
    st.none(),
    st.lists(st.integers()),
    st.dictionaries(st.text(), st.integers())
))
@settings(max_examples=1000)
def test_is_re_compilable_correctness(obj):
    result = is_re_compilable(obj)

    if result:
        try:
            re.compile(obj)
        except TypeError:
            assert False, f"is_re_compilable({obj!r}) returned True but re.compile() raised TypeError"
    else:
        try:
            re.compile(obj)
            assert False, f"is_re_compilable({obj!r}) returned False but re.compile() succeeded"
        except TypeError:
            pass
```

**Failing input**: `'\\'` (single backslash)

## Reproducing the Bug

```python
from pandas.api.types import is_re_compilable

is_re_compilable('\\')
```

```
re.PatternError: bad escape (end of pattern) at position 0
```

Other failing inputs: `'['`, `'(?P<'`, `'*'`

## Why This Is A Bug

The function's docstring states it should "Check if the object can be compiled into a regex pattern instance" and return a boolean. Users expect a predicate function to return True/False, not crash. The function only catches `TypeError` but `re.compile()` also raises `re.PatternError` for invalid regex syntax.

## Fix

```diff
--- a/pandas/core/dtypes/inference.py
+++ b/pandas/core/dtypes/inference.py
@@ -185,7 +185,7 @@ def is_re_compilable(obj) -> bool:
     """
     try:
         re.compile(obj)
-    except TypeError:
+    except (TypeError, re.error):
         return False
     else:
         return True
```