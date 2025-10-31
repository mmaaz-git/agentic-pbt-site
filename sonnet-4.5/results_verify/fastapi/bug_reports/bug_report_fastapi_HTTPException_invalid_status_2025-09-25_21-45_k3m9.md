# Bug Report: FastAPI HTTPException Invalid Status Code

**Target**: `fastapi.exceptions.HTTPException`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

HTTPException crashes with ValueError when initialized with a non-standard HTTP status code (e.g., 104, 599) and no detail parameter, despite accepting any integer as status_code according to its type signature.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi.exceptions import HTTPException


@given(st.integers(min_value=100, max_value=599))
def test_http_exception_without_detail(status_code):
    exc = HTTPException(status_code=status_code)
    assert exc.status_code == status_code
```

**Failing input**: `status_code=104`

## Reproducing the Bug

```python
from fastapi.exceptions import HTTPException

exc = HTTPException(status_code=104)
```

Raises:
```
ValueError: 104 is not a valid HTTPStatus
```

However, providing a detail works fine:
```python
exc = HTTPException(status_code=104, detail="Custom detail")
```

## Why This Is A Bug

The HTTPException constructor accepts `status_code: int` according to its type signature, suggesting any integer should be valid. However, when `detail` is None (the default), the underlying Starlette implementation tries to look up the HTTP status phrase using `http.HTTPStatus(status_code).phrase`, which only works for standard HTTP status codes.

This creates an inconsistent API:
- `HTTPException(status_code=104, detail="error")` works
- `HTTPException(status_code=104)` crashes

Users may legitimately want to use custom status codes (e.g., 599 for custom application errors) without providing a detail, or they may simply make a typo (e.g., 440 instead of 404) and get a confusing ValueError instead of using their custom status code.

## Fix

```diff
--- a/fastapi/exceptions.py
+++ b/fastapi/exceptions.py
@@ -62,5 +62,11 @@ class HTTPException(StarletteHTTPException):
         ] = None,
     ) -> None:
-        super().__init__(status_code=status_code, detail=detail, headers=headers)
+        # If detail is None and status_code is not a valid HTTPStatus,
+        # provide a default detail to avoid ValueError
+        if detail is None:
+            try:
+                super().__init__(status_code=status_code, detail=detail, headers=headers)
+            except ValueError:
+                super().__init__(status_code=status_code, detail=f"HTTP {status_code}", headers=headers)
+        else:
+            super().__init__(status_code=status_code, detail=detail, headers=headers)
```

Alternatively, the signature could be updated to only accept valid HTTP status codes, but this would be a breaking change and less flexible for custom status codes.