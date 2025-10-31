# Bug Report: fastapi.security API Key Whitespace Validation

**Target**: `fastapi.security.api_key.APIKeyBase.check_api_key`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `check_api_key` method incorrectly accepts whitespace-only API keys (e.g., `" "`, `"\t"`) as valid, while rejecting empty strings. This inconsistent validation creates a security/logic gap where whitespace strings bypass authentication checks.

## Property-Based Test

```python
from unittest.mock import Mock

import pytest
from fastapi.security.api_key import APIKeyHeader
from hypothesis import given, settings, strategies as st
from starlette.requests import Request


@pytest.mark.asyncio
@given(st.sampled_from([" ", "  ", "\t", "\n", "\r", "   ", " \t ", "\t\n", " \n\r\t "]))
@settings(max_examples=20, deadline=None)
async def test_api_key_whitespace_only_should_be_rejected(whitespace_key):
    api_key = APIKeyHeader(name="X-API-Key", auto_error=False)
    request = Mock(spec=Request)
    request.headers = {"X-API-Key": whitespace_key}

    result = await api_key(request)

    assert result is None or result.strip() != "", \
        f"Whitespace-only API key {whitespace_key!r} should be rejected, but got {result!r}"
```

**Failing input**: `whitespace_key=' '`

## Reproducing the Bug

```python
from unittest.mock import Mock

import pytest
from fastapi.security.api_key import APIKeyHeader
from starlette.exceptions import HTTPException
from starlette.requests import Request


@pytest.mark.asyncio
async def test_empty_string_is_rejected():
    api_key = APIKeyHeader(name="X-API-Key")
    request = Mock(spec=Request)
    request.headers = {"X-API-Key": ""}

    with pytest.raises(HTTPException) as exc:
        await api_key(request)

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_whitespace_is_accepted():
    api_key = APIKeyHeader(name="X-API-Key")
    request = Mock(spec=Request)
    request.headers = {"X-API-Key": " "}

    result = await api_key(request)
    assert result == " "
```

## Why This Is A Bug

The `check_api_key` method uses `if not api_key` to validate API keys, which only rejects falsy values. In Python, empty string `""` is falsy (correctly rejected), but whitespace strings like `" "` are truthy (incorrectly accepted).

This creates inconsistent behavior:
- `""` → Rejected with 403 error
- `" "` → Accepted as valid

HTTP headers and query parameters can legitimately contain whitespace (they're not automatically trimmed), so a client could send `X-API-Key: ` (just spaces) and bypass the authentication check.

While downstream validation would likely reject these whitespace keys, the inconsistency violates the principle of least surprise and could cause security issues if code checks "is result not None" rather than validating the actual key value.

## Fix

```diff
--- a/fastapi/security/api_key.py
+++ b/fastapi/security/api_key.py
@@ -11,7 +11,7 @@ class APIKeyBase(SecurityBase):
 class APIKeyBase(SecurityBase):
     @staticmethod
     def check_api_key(api_key: Optional[str], auto_error: bool) -> Optional[str]:
-        if not api_key:
+        if not api_key or not api_key.strip():
             if auto_error:
                 raise HTTPException(
                     status_code=HTTP_403_FORBIDDEN, detail="Not authenticated"
```