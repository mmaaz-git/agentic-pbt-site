# Bug Report: FastAPI HTTPException Crashes with Non-Standard HTTP Status Codes

**Target**: `fastapi.exceptions.HTTPException`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

HTTPException crashes with ValueError when initialized with non-standard HTTP status codes (e.g., 104, 599) without a detail parameter, despite its type signature accepting any integer as status_code.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi.exceptions import HTTPException


@given(st.integers(min_value=100, max_value=599))
def test_http_exception_without_detail(status_code):
    exc = HTTPException(status_code=status_code)
    assert exc.status_code == status_code


if __name__ == "__main__":
    test_http_exception_without_detail()
```

<details>

<summary>
**Failing input**: `status_code=104`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 12, in <module>
    test_http_exception_without_detail()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 6, in test_http_exception_without_detail
    def test_http_exception_without_detail(status_code):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 7, in test_http_exception_without_detail
    exc = HTTPException(status_code=status_code)
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/exceptions.py", line 65, in __init__
    super().__init__(status_code=status_code, detail=detail, headers=headers)
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/starlette/exceptions.py", line 10, in __init__
    detail = http.HTTPStatus(status_code).phrase
             ~~~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/enum.py", line 726, in __call__
    return cls.__new__(cls, value)
           ~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/enum.py", line 1199, in __new__
    raise ve_exc
ValueError: 104 is not a valid HTTPStatus
Falsifying example: test_http_exception_without_detail(
    status_code=104,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/enum.py:1163
```
</details>

## Reproducing the Bug

```python
from fastapi.exceptions import HTTPException

# This will crash with a ValueError
try:
    exc = HTTPException(status_code=104)
    print(f"Success: Created HTTPException with status_code={exc.status_code}")
except ValueError as e:
    print(f"Error: {e}")

# This works fine when providing a detail
try:
    exc = HTTPException(status_code=104, detail="Custom detail")
    print(f"Success: Created HTTPException with status_code={exc.status_code} and detail='{exc.detail}'")
except ValueError as e:
    print(f"Error: {e}")
```

<details>

<summary>
ValueError when creating HTTPException without detail for non-standard status code
</summary>
```
Error: 104 is not a valid HTTPStatus
Success: Created HTTPException with status_code=104 and detail='Custom detail'
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **Type Signature Contradiction**: The FastAPI HTTPException constructor declares `status_code: int` in its type signature (fastapi/exceptions.py:39-46), which indicates that any integer value should be accepted. However, the implementation crashes for many valid integer values when detail is None.

2. **Inconsistent API Behavior**: The same status code produces different results based on whether detail is provided:
   - `HTTPException(status_code=104, detail="error")` → Works correctly
   - `HTTPException(status_code=104, detail=None)` → Crashes with ValueError
   - `HTTPException(status_code=104)` → Crashes with ValueError (detail defaults to None)

3. **Documentation Mismatch**: The documentation states the parameter is "HTTP status code to send to the client" without any restrictions on which codes are valid. Neither FastAPI's documentation nor the docstring mentions that only standard HTTP status codes work without a detail parameter.

4. **Valid Use Cases Blocked**: HTTP specifications allow custom status codes within valid ranges (100-599). Applications legitimately use custom codes like 599 for application-specific errors. Additionally, simple typos (e.g., 440 instead of 404) produce confusing ValueError messages rather than just using the provided code.

5. **Root Cause**: The issue stems from Starlette's HTTPException implementation (starlette/exceptions.py:10) which attempts to look up the status phrase using `http.HTTPStatus(status_code).phrase` when detail is None. Python's HTTPStatus enum only contains standard codes, causing a ValueError for any non-standard code.

## Relevant Context

The bug originates in Starlette's exception handling, not FastAPI itself. FastAPI's HTTPException inherits from StarletteHTTPException and simply passes all parameters through without modification (fastapi/exceptions.py:65).

The problematic line in Starlette (starlette/exceptions.py:10):
```python
if detail is None:
    detail = http.HTTPStatus(status_code).phrase
```

This assumes all status codes exist in Python's HTTPStatus enum, which only includes standard codes defined in various HTTP RFCs. Custom status codes are valid per HTTP specifications but aren't in this enum.

Relevant links:
- FastAPI HTTPException source: https://github.com/tiangolo/fastapi/blob/master/fastapi/exceptions.py
- Starlette HTTPException source: https://github.com/encode/starlette/blob/master/starlette/exceptions.py
- FastAPI error handling docs: https://fastapi.tiangolo.com/tutorial/handling-errors/

## Proposed Fix

The fix should handle non-standard status codes gracefully when detail is None. Here's a patch for FastAPI that works around the Starlette limitation:

```diff
--- a/fastapi/exceptions.py
+++ b/fastapi/exceptions.py
@@ -1,4 +1,5 @@
 from typing import Any, Dict, Optional, Sequence, Type, Union
+import http

 from pydantic import BaseModel, create_model
 from starlette.exceptions import HTTPException as StarletteHTTPException
@@ -62,7 +63,17 @@ class HTTPException(StarletteHTTPException):
             ),
         ] = None,
     ) -> None:
-        super().__init__(status_code=status_code, detail=detail, headers=headers)
+        # If detail is None and status_code is not a standard HTTP status,
+        # provide a default detail to avoid ValueError from Starlette
+        if detail is None:
+            try:
+                # Try to get the standard phrase
+                http.HTTPStatus(status_code)
+                # If successful, use Starlette's default behavior
+                super().__init__(status_code=status_code, detail=detail, headers=headers)
+            except ValueError:
+                # For non-standard codes, provide a generic detail
+                super().__init__(status_code=status_code, detail=f"HTTP {status_code}", headers=headers)
+        else:
+            super().__init__(status_code=status_code, detail=detail, headers=headers)


 class WebSocketException(StarletteWebSocketException):
```