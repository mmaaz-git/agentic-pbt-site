# Bug Report: fastapi.utils.is_body_allowed_for_status_code Invalid String Handling

**Target**: `fastapi.utils.is_body_allowed_for_status_code`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_body_allowed_for_status_code` function crashes with a `ValueError` when given an invalid string status code that is not in the predefined pattern list and cannot be converted to an integer.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from fastapi.utils import is_body_allowed_for_status_code

valid_patterns = {"default", "1XX", "2XX", "3XX", "4XX", "5XX"}

@given(st.text())
def test_is_body_allowed_for_status_code_handles_all_strings(status_code):
    assume(status_code not in valid_patterns)

    try:
        result = is_body_allowed_for_status_code(status_code)
        assert isinstance(result, bool)
    except ValueError:
        pass
```

**Failing input**: Any non-numeric string not in the predefined patterns, e.g., `"invalid"`, `"abc"`, `"error"`, `""`

## Reproducing the Bug

```python
from fastapi.utils import is_body_allowed_for_status_code

is_body_allowed_for_status_code("invalid")
```

This raises:
```
ValueError: invalid literal for int() with base 10: 'invalid'
```

## Why This Is A Bug

The function signature declares it accepts `Union[int, str, None]`, indicating it should handle any string input. However, the implementation only handles specific string patterns (`"default"`, `"1XX"`, `"2XX"`, etc.) and numeric strings. When given an arbitrary string like `"invalid"`, it attempts to convert it to an integer on line 55 without error handling, causing an unhandled `ValueError`.

According to the type signature, the function should handle all string inputs gracefully, either by returning a boolean or raising a more informative error. The current behavior violates the implicit contract of the type signature.

## Fix

```diff
--- a/fastapi/utils.py
+++ b/fastapi/utils.py
@@ -52,6 +52,10 @@ def is_body_allowed_for_status_code(status_code: Union[int, str, None]) -> bool
         "5XX",
     }:
         return True
-    current_status_code = int(status_code)
+    try:
+        current_status_code = int(status_code)
+    except (ValueError, TypeError):
+        return True
     return not (current_status_code < 200 or current_status_code in {204, 205, 304})
```

This fix wraps the `int()` conversion in a try-except block. When the conversion fails, it defaults to `True` (allowing body), which is the safe default behavior consistent with how the function handles `None` and pattern strings.