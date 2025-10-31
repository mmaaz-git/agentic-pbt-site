# Bug Report: FastAPI ResponseValidationError Grammar Error

**Target**: `fastapi.exceptions.ResponseValidationError.__str__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ResponseValidationError.__str__` method always uses plural "errors" even when there is exactly one validation error, resulting in grammatically incorrect output like "1 validation errors:" instead of "1 validation error:".

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from fastapi.exceptions import ResponseValidationError


@given(st.lists(st.text(), min_size=1, max_size=1))
def test_single_error_uses_singular_form(errors):
    exc = ResponseValidationError(errors)
    result = str(exc)

    assert "1 validation error:" in result, \
        f"Expected singular 'error', got: {result.split(chr(10))[0]}"
```

**Failing input**: `['any single error message']`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

from fastapi.exceptions import ResponseValidationError

exc = ResponseValidationError(["field validation failed"])
print(str(exc))
```

**Actual output:**
```
1 validation errors:
  field validation failed
```

**Expected output:**
```
1 validation error:
  field validation failed
```

## Why This Is A Bug

This violates basic English grammar rules. When there is exactly one error, the singular form "error" should be used, not the plural "errors". This is a UX/display issue that makes error messages look unprofessional and can confuse users.

## Fix

```diff
--- a/fastapi/exceptions.py
+++ b/fastapi/exceptions.py
@@ -170,7 +170,8 @@ class ResponseValidationError(ValidationException):
         self.body = body

     def __str__(self) -> str:
-        message = f"{len(self._errors)} validation errors:\n"
+        count = len(self._errors)
+        message = f"{count} validation {'error' if count == 1 else 'errors'}:\n"
         for err in self._errors:
             message += f"  {err}\n"
         return message
```