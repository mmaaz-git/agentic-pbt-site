# Bug Report: FastAPI is_body_allowed_for_status_code ValueError on Invalid Strings

**Target**: `fastapi.utils.is_body_allowed_for_status_code`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_body_allowed_for_status_code` function crashes with `ValueError` when given string status codes that are not in the predefined set and cannot be converted to integers, violating the function's type signature which accepts `Union[int, str, None]`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi.utils import is_body_allowed_for_status_code

@given(st.one_of(st.none(), st.integers(), st.text()))
def test_is_body_allowed_no_crash(status_code):
    result = is_body_allowed_for_status_code(status_code)
    assert isinstance(result, bool)
```

**Failing input**: `"abc"`, `"1.5"`, `"200.0"`, `""`, `"200 OK"`, or any non-numeric string

## Reproducing the Bug

```python
from fastapi.utils import is_body_allowed_for_status_code

try:
    result = is_body_allowed_for_status_code("abc")
except ValueError as e:
    print(f"ValueError: {e}")

try:
    result = is_body_allowed_for_status_code("200.0")
except ValueError as e:
    print(f"ValueError: {e}")
```

## Why This Is A Bug

The function's type signature explicitly accepts `Union[int, str, None]`, indicating it should handle arbitrary strings. While it correctly handles specific OpenAPI status code patterns ("1XX", "2XX", etc.), it assumes all other strings are valid integers, causing crashes on:

1. Invalid status codes from user-defined responses
2. Malformed status codes in OpenAPI schemas
3. Any string input that's not a predefined pattern or numeric

The function should either:
- Validate string inputs and handle invalid cases gracefully
- Raise a more specific exception with a clear error message
- Document that only numeric strings and specific patterns are accepted

## Fix

```diff
 def is_body_allowed_for_status_code(status_code: Union[int, str, None]) -> bool:
     if status_code is None:
         return True
     if status_code in {
         "default",
         "1XX",
         "2XX",
         "3XX",
         "4XX",
         "5XX",
     }:
         return True
-    current_status_code = int(status_code)
+    try:
+        current_status_code = int(status_code)
+    except (ValueError, TypeError):
+        raise ValueError(
+            f"Invalid status code: {status_code!r}. "
+            f"Expected an integer, valid integer string, or one of: "
+            f"'default', '1XX', '2XX', '3XX', '4XX', '5XX'"
+        )
     return not (current_status_code < 200 or current_status_code in {204, 205, 304})
```