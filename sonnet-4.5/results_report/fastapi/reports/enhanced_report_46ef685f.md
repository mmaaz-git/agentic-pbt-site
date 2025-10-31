# Bug Report: FastAPI ResponseValidationError Incorrect Grammar for Singular Error

**Target**: `fastapi.exceptions.ResponseValidationError.__str__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ResponseValidationError.__str__` method incorrectly uses plural "errors" when displaying exactly one validation error, resulting in grammatically incorrect output like "1 validation errors:" instead of "1 validation error:".

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

if __name__ == "__main__":
    test_single_error_uses_singular_form()
```

<details>

<summary>
**Failing input**: `['']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 17, in <module>
    test_single_error_uses_singular_form()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 9, in test_single_error_uses_singular_form
    def test_single_error_uses_singular_form(errors):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 13, in test_single_error_uses_singular_form
    assert "1 validation error:" in result, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected singular 'error', got: 1 validation errors:
Falsifying example: test_single_error_uses_singular_form(
    errors=[''],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/38/hypo.py:14
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

from fastapi.exceptions import ResponseValidationError

# Test with single error
exc = ResponseValidationError(["field validation failed"])
print("Single error case:")
print(str(exc))
print()

# Test with multiple errors
exc_multiple = ResponseValidationError(["field1 validation failed", "field2 validation failed"])
print("Multiple errors case:")
print(str(exc_multiple))
print()

# Test with zero errors (edge case)
exc_zero = ResponseValidationError([])
print("Zero errors case:")
print(str(exc_zero))
```

<details>

<summary>
Demonstrates grammatically incorrect output for single error case
</summary>
```
Single error case:
1 validation errors:
  field validation failed


Multiple errors case:
2 validation errors:
  field1 validation failed
  field2 validation failed


Zero errors case:
0 validation errors:

```
</details>

## Why This Is A Bug

This violates standard English grammar rules where singular nouns require singular forms ("1 validation error" not "1 validation errors"). The issue occurs because the `__str__` method in `ResponseValidationError` (lines 172-176 of `/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/fastapi/exceptions.py`) always hardcodes "errors" in plural form regardless of the actual count. This is inconsistent with proper grammar conventions and differs from how similar libraries like Pydantic handle error formatting, where they correctly use "1 validation error" for single errors and "X validation errors" for multiple errors.

## Relevant Context

The bug is located in the `ResponseValidationError` class which inherits from `ValidationException`. Interestingly, other similar exception classes in the same file (`RequestValidationError` at line 157 and `WebSocketRequestValidationError` at line 163) do not override the `__str__` method, making `ResponseValidationError` unique in having this custom string representation.

The issue affects all cases where exactly one validation error is present, making error messages appear unprofessional. While the functionality remains intact and messages are still comprehensible, this grammatical inconsistency detracts from the overall quality and professionalism of error reporting in FastAPI applications.

FastAPI documentation: https://fastapi.tiangolo.com/
Source code location: `/fastapi/exceptions.py` lines 172-176

## Proposed Fix

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