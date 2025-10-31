# Bug Report: FastAPI Security API Key Whitespace Validation Inconsistency

**Target**: `fastapi.security.api_key.APIKeyBase.check_api_key`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `check_api_key` method in FastAPI's security module incorrectly accepts whitespace-only API keys (e.g., `" "`, `"\t"`, `"\n"`) as valid authentication tokens, while correctly rejecting empty strings (`""`). This creates an inconsistent validation behavior where meaningless whitespace strings bypass the authentication check at the framework level.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test that discovered the FastAPI API Key whitespace validation bug."""

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


if __name__ == "__main__":
    # Run with pytest
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "-s", "--tb=short"],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    sys.exit(result.returncode)
```

<details>

<summary>
**Failing input**: `whitespace_key=' '`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/33
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_api_key_whitespace_only_should_be_rejected FAILED

=================================== FAILURES ===================================
_______________ test_api_key_whitespace_only_should_be_rejected ________________
hypo.py:13: in test_api_key_whitespace_only_should_be_rejected
    @given(st.sampled_from([" ", "  ", "\t", "\n", "\r", "   ", " \t ", "\t\n", " \n\r\t "]))
                   ^^^
/home/npc/miniconda/lib/python3.13/site-packages/pytest_asyncio/plugin.py:721: in inner
    runner.run(coro, context=context)
/home/npc/miniconda/lib/python3.13/asyncio/runners.py:118: in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/asyncio/base_events.py:725: in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
hypo.py:22: in test_api_key_whitespace_only_should_be_rejected
    assert result is None or result.strip() != "", \
E   AssertionError: Whitespace-only API key ' ' should be rejected, but got ' '
E   assert (' ' is None or '' != '')
E    +  where '' = <built-in method strip of str object at 0x8bc1d8>()
E    +    where <built-in method strip of str object at 0x8bc1d8> = ' '.strip
E   Falsifying example: test_api_key_whitespace_only_should_be_rejected(
E       whitespace_key=' ',
E   )
=========================== short test summary info ============================
FAILED hypo.py::test_api_key_whitespace_only_should_be_rejected - AssertionEr...
============================== 1 failed in 0.38s ===============================
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of FastAPI API Key whitespace validation bug."""

import asyncio
from unittest.mock import Mock

from fastapi.security.api_key import APIKeyHeader
from starlette.exceptions import HTTPException
from starlette.requests import Request


async def test_empty_string_rejected():
    """Test that empty string is rejected (this works correctly)."""
    api_key = APIKeyHeader(name="X-API-Key")
    request = Mock(spec=Request)
    request.headers = {"X-API-Key": ""}

    try:
        result = await api_key(request)
        print(f"ERROR: Empty string should be rejected but got: {result!r}")
    except HTTPException as e:
        print(f"✓ Empty string correctly rejected with HTTP {e.status_code}: {e.detail}")


async def test_whitespace_accepted():
    """Test that whitespace is incorrectly accepted."""
    api_key = APIKeyHeader(name="X-API-Key")
    request = Mock(spec=Request)
    request.headers = {"X-API-Key": " "}

    try:
        result = await api_key(request)
        print(f"✗ BUG: Whitespace-only string ' ' incorrectly accepted, returned: {result!r}")
    except HTTPException as e:
        print(f"Whitespace rejected with HTTP {e.status_code}: {e.detail}")


async def test_various_whitespace():
    """Test various whitespace characters."""
    api_key = APIKeyHeader(name="X-API-Key", auto_error=False)

    whitespace_chars = [
        (" ", "single space"),
        ("  ", "double space"),
        ("\t", "tab"),
        ("\n", "newline"),
        ("\r", "carriage return"),
        ("   ", "triple space"),
        (" \t ", "space-tab-space"),
        ("\t\n", "tab-newline"),
        (" \n\r\t ", "mixed whitespace"),
    ]

    print("\nTesting various whitespace characters (auto_error=False):")
    for ws, description in whitespace_chars:
        request = Mock(spec=Request)
        request.headers = {"X-API-Key": ws}
        result = await api_key(request)
        status = "✗ ACCEPTED" if result else "✓ REJECTED"
        print(f"  {description:20} ({repr(ws):10}): {status} - returned {result!r}")


async def test_missing_header():
    """Test missing header behavior."""
    api_key = APIKeyHeader(name="X-API-Key", auto_error=False)
    request = Mock(spec=Request)
    request.headers = {}

    result = await api_key(request)
    print(f"\n✓ Missing header correctly returns: {result!r}")


async def test_valid_api_key():
    """Test valid API key is accepted."""
    api_key = APIKeyHeader(name="X-API-Key")
    request = Mock(spec=Request)
    request.headers = {"X-API-Key": "valid-api-key-123"}

    result = await api_key(request)
    print(f"✓ Valid API key correctly accepted: {result!r}")


async def main():
    print("=== FastAPI API Key Whitespace Validation Bug Demonstration ===\n")

    await test_empty_string_rejected()
    await test_whitespace_accepted()
    await test_various_whitespace()
    await test_missing_header()
    await test_valid_api_key()

    print("\n=== Summary ===")
    print("The bug is in APIKeyBase.check_api_key() at line 14:")
    print("  if not api_key:")
    print("This check rejects falsy values (None, '') but accepts truthy whitespace (' ', '\\t', etc.)")
    print("Fix: Change to: if not api_key or not api_key.strip():")


if __name__ == "__main__":
    asyncio.run(main())
```

<details>

<summary>
BUG: Whitespace-only API keys are incorrectly accepted as valid
</summary>
```
=== FastAPI API Key Whitespace Validation Bug Demonstration ===

✓ Empty string correctly rejected with HTTP 403: Not authenticated
✗ BUG: Whitespace-only string ' ' incorrectly accepted, returned: ' '

Testing various whitespace characters (auto_error=False):
  single space         (' '       ): ✗ ACCEPTED - returned ' '
  double space         ('  '      ): ✗ ACCEPTED - returned '  '
  tab                  ('\t'      ): ✗ ACCEPTED - returned '\t'
  newline              ('\n'      ): ✗ ACCEPTED - returned '\n'
  carriage return      ('\r'      ): ✗ ACCEPTED - returned '\r'
  triple space         ('   '     ): ✗ ACCEPTED - returned '   '
  space-tab-space      (' \t '    ): ✗ ACCEPTED - returned ' \t '
  tab-newline          ('\t\n'    ): ✗ ACCEPTED - returned '\t\n'
  mixed whitespace     (' \n\r\t '): ✗ ACCEPTED - returned ' \n\r\t '

✓ Missing header correctly returns: None
✓ Valid API key correctly accepted: 'valid-api-key-123'

=== Summary ===
The bug is in APIKeyBase.check_api_key() at line 14:
  if not api_key:
This check rejects falsy values (None, '') but accepts truthy whitespace (' ', '\t', etc.)
Fix: Change to: if not api_key or not api_key.strip():
```
</details>

## Why This Is A Bug

This behavior violates the expected contract of an authentication system in several critical ways:

1. **Inconsistent Validation Logic**: The method correctly rejects empty strings (`""`) with an HTTP 403 error but accepts whitespace-only strings (`" "`, `"\t"`, etc.) as valid API keys. This inconsistency stems from Python's truthiness evaluation where empty strings are falsy but non-empty whitespace strings are truthy.

2. **Security Principle Violation**: Authentication systems should follow the principle of "fail closed" - rejecting questionable or meaningless input. A whitespace-only string has no semantic meaning as an authentication token and should be rejected at the framework level.

3. **Documentation Ambiguity**: The documentation states that when `auto_error=True` (default), the system will raise an error "when the header is not available" or "not provided". A header containing only whitespace is effectively "not provided" from a security perspective, yet the current implementation treats it as valid.

4. **HTTP Protocol Consideration**: HTTP headers are not automatically trimmed by the protocol specification. A client sending `X-API-Key: ` (with trailing spaces) or `X-API-Key:\t` will have those whitespace characters preserved and passed to the application. The current implementation would accept these as valid authentication tokens.

5. **Downstream Impact**: While application code could add additional validation, developers reasonably expect the authentication layer to reject meaningless values. Code that checks `if api_key is not None:` would incorrectly assume a whitespace key is valid.

## Relevant Context

The bug affects all three API key authentication methods in FastAPI:
- `APIKeyHeader` (tested above) - Extracts API key from HTTP headers
- `APIKeyQuery` - Extracts API key from query parameters
- `APIKeyCookie` - Extracts API key from cookies

All three classes inherit from `APIKeyBase` and use the same flawed `check_api_key` static method at line 14 of `/fastapi/security/api_key.py`:

```python
@staticmethod
def check_api_key(api_key: Optional[str], auto_error: bool) -> Optional[str]:
    if not api_key:  # <-- Bug is here
        if auto_error:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN, detail="Not authenticated"
            )
        return None
    return api_key
```

This is a common pattern in authentication libraries where blank values should be rejected. For comparison, most authentication frameworks treat both empty and whitespace-only values as invalid tokens.

FastAPI documentation: https://fastapi.tiangolo.com/tutorial/security/api-key/
Source code: https://github.com/tiangolo/fastapi/blob/master/fastapi/security/api_key.py

## Proposed Fix

```diff
--- a/fastapi/security/api_key.py
+++ b/fastapi/security/api_key.py
@@ -11,7 +11,7 @@ from typing_extensions import Annotated, Doc
 class APIKeyBase(SecurityBase):
     @staticmethod
     def check_api_key(api_key: Optional[str], auto_error: bool) -> Optional[str]:
-        if not api_key:
+        if not api_key or not api_key.strip():
             if auto_error:
                 raise HTTPException(
                     status_code=HTTP_403_FORBIDDEN, detail="Not authenticated"
```