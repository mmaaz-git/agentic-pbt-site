# Bug Report: pandas.api.types.is_re_compilable Raises Exception Instead of Returning False

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `is_re_compilable` function is documented to return a boolean indicating whether an object can be compiled as a regex pattern. However, it raises `re.PatternError` for invalid regex strings instead of returning `False`.

## Property-Based Test

```python
import pandas.api.types as pat
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=1, max_size=100))
@settings(max_examples=200)
def test_is_re_compilable_on_strings(s):
    result = pat.is_re_compilable(s)
    assert isinstance(result, bool), f"is_re_compilable should return bool"
```

**Failing input**: `'['`

## Reproducing the Bug

```python
import pandas.api.types as pat

invalid_regex_patterns = ['[', '?', '*', '(unclosed']

for pattern in invalid_regex_patterns:
    result = pat.is_re_compilable(pattern)
```

Running this code raises:
```
re.PatternError: unterminated character set at position 0
```

## Why This Is A Bug

The function's docstring explicitly states:
> "Check if the object can be compiled into a regex pattern instance."
>
> Returns: bool - Whether `obj` can be compiled as a regex pattern.

The function promises to return a boolean, but instead raises an exception for invalid regex patterns. This violates the API contract. A user calling this function to check if a string is a valid regex will get an unexpected exception instead of the documented boolean return value.

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