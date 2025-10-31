# Bug Report: FastAPI utils.is_body_allowed_for_status_code ValueError on Non-Numeric Strings

**Target**: `fastapi.utils.is_body_allowed_for_status_code`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_body_allowed_for_status_code` function crashes with `ValueError` when given string status codes that cannot be converted to integers, violating its type signature `Union[int, str, None]` which explicitly promises to accept any string.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi.utils import is_body_allowed_for_status_code

@given(st.one_of(st.none(), st.integers(), st.text()))
def test_is_body_allowed_no_crash(status_code):
    result = is_body_allowed_for_status_code(status_code)
    assert isinstance(result, bool)

if __name__ == "__main__":
    test_is_body_allowed_no_crash()
```

<details>

<summary>
**Failing input**: `''`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 10, in <module>
    test_is_body_allowed_no_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 5, in test_is_body_allowed_no_crash
    def test_is_body_allowed_no_crash(status_code):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 6, in test_is_body_allowed_no_crash
    result = is_body_allowed_for_status_code(status_code)
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/utils.py", line 55, in is_body_allowed_for_status_code
    current_status_code = int(status_code)
ValueError: invalid literal for int() with base 10: ''
Falsifying example: test_is_body_allowed_no_crash(
    status_code='',
)
```
</details>

## Reproducing the Bug

```python
from fastapi.utils import is_body_allowed_for_status_code

# Test various string inputs that should be accepted based on the type signature
test_cases = [
    "abc",           # Non-numeric string
    "200.0",         # Float-like string
    "",              # Empty string
    "200 OK",        # Status code with description
    "1.5",           # Decimal string
    "not_a_number",  # Text string
]

print("Testing is_body_allowed_for_status_code with various string inputs:\n")

for status_code in test_cases:
    try:
        result = is_body_allowed_for_status_code(status_code)
        print(f"  {repr(status_code)}: {result}")
    except ValueError as e:
        print(f"  {repr(status_code)}: ValueError - {e}")
    except Exception as e:
        print(f"  {repr(status_code)}: {type(e).__name__} - {e}")

# Also test valid inputs to show they work correctly
print("\nValid inputs that work correctly:")
valid_cases = [
    200,        # Integer
    "200",      # Numeric string
    "2XX",      # OpenAPI pattern
    "default",  # OpenAPI default
    None,       # None value
]

for status_code in valid_cases:
    try:
        result = is_body_allowed_for_status_code(status_code)
        print(f"  {repr(status_code)}: {result}")
    except Exception as e:
        print(f"  {repr(status_code)}: {type(e).__name__} - {e}")
```

<details>

<summary>
ValueError crashes on non-numeric strings
</summary>
```
Testing is_body_allowed_for_status_code with various string inputs:

  'abc': ValueError - invalid literal for int() with base 10: 'abc'
  '200.0': ValueError - invalid literal for int() with base 10: '200.0'
  '': ValueError - invalid literal for int() with base 10: ''
  '200 OK': ValueError - invalid literal for int() with base 10: '200 OK'
  '1.5': ValueError - invalid literal for int() with base 10: '1.5'
  'not_a_number': ValueError - invalid literal for int() with base 10: 'not_a_number'

Valid inputs that work correctly:
  200: True
  '200': True
  '2XX': True
  'default': True
  None: True
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Type Contract Violation**: The function signature explicitly accepts `Union[int, str, None]`, promising to handle any string type. However, it crashes on non-numeric strings that aren't in the predefined OpenAPI pattern set (`"default"`, `"1XX"`, `"2XX"`, `"3XX"`, `"4XX"`, `"5XX"`).

2. **FastAPI Usage Context**: FastAPI's `APIRoute` class explicitly accepts `responses: Dict[Union[int, str], Dict[str, Any]]`, allowing arbitrary string status codes. The function is called with these user-provided status codes in `routing.py` (lines 544 and 644), potentially causing production crashes.

3. **Unchecked Assumption**: Line 55 of `utils.py` unconditionally calls `int(status_code)` on any string not in the predefined set, without error handling for invalid conversions. This assumes all non-pattern strings are numeric, which contradicts the type signature.

4. **User Impact**: Users can legitimately provide custom string status codes in their API responses (e.g., custom error codes, vendor-specific status codes), and the function will crash rather than handling them gracefully.

## Relevant Context

The function is located at `/lib/python3.13/site-packages/fastapi/utils.py` lines 42-56. It determines whether HTTP response bodies are allowed for given status codes based on HTTP standards (status codes 204, 205, 304, and 1xx codes should not have bodies).

The function is used in critical paths:
- `fastapi/routing.py:544` - Validates response models for status codes
- `fastapi/routing.py:644` - Validates additional responses in route definitions
- `fastapi/exception_handlers.py` - Handles response bodies for exceptions

FastAPI documentation: https://fastapi.tiangolo.com/tutorial/response-status-code/
OpenAPI specification for patterned fields: https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.1.0.md#patterned-fields-1

## Proposed Fix

```diff
 def is_body_allowed_for_status_code(status_code: Union[int, str, None]) -> bool:
     if status_code is None:
         return True
     # Ref: https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.1.0.md#patterned-fields-1
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
+        # For non-standard status codes, default to allowing body
+        # since we cannot determine the appropriate behavior
+        return True
     return not (current_status_code < 200 or current_status_code in {204, 205, 304})
```