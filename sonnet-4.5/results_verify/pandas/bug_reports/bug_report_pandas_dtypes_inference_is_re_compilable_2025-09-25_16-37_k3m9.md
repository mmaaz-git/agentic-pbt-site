# Bug Report: pandas.core.dtypes.inference.is_re_compilable Raises Exception Instead of Returning False

**Target**: `pandas.core.dtypes.inference.is_re_compilable`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `is_re_compilable` function is documented to return a boolean indicating whether an object can be compiled as a regex pattern. However, when given invalid regex patterns (like `")"` or `"?"`), it raises a `PatternError` exception instead of returning `False`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.core.dtypes import inference
import re

@given(st.text(min_size=0, max_size=100))
def test_is_re_compilable_on_regex_patterns(pattern):
    try:
        re.compile(pattern)
        result = inference.is_re_compilable(pattern)
        assert result is True
    except re.error:
        result = inference.is_re_compilable(pattern)
        assert result is False
```

**Failing input**: `pattern=')'`

## Reproducing the Bug

```python
from pandas.core.dtypes.inference import is_re_compilable

result = is_re_compilable(")")
```

Expected: `False`
Actual: Raises `PatternError: unbalanced parenthesis at position 0`

## Why This Is A Bug

The function's documented return type is `bool`, and its docstring states it should check "Whether `obj` can be compiled as a regex pattern." When a pattern cannot be compiled, the function should return `False`, not raise an exception. This violates the API contract.

The function currently only catches `TypeError`, but `re.compile()` can also raise `re.error` (or `PatternError` in Python 3.13+) for invalid regex syntax.

## Fix

```diff
--- a/pandas/core/dtypes/inference.py
+++ b/pandas/core/dtypes/inference.py
@@ -184,7 +184,7 @@ def is_re_compilable(obj) -> bool:
     """
     try:
         re.compile(obj)
-    except TypeError:
+    except (TypeError, re.error):
         return False
     else:
         return True
```