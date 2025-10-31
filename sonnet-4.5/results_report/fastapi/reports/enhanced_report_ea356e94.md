# Bug Report: fastapi.security.api_key Accepts Whitespace-Only API Keys

**Target**: `fastapi.security.api_key.APIKeyBase.check_api_key`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `check_api_key` method in FastAPI's security module accepts whitespace-only strings as valid API keys while rejecting empty strings, creating an inconsistent validation behavior that could lead to security vulnerabilities.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi.security.api_key import APIKeyBase


@given(whitespace=st.text(alphabet=' \t\r\n', min_size=1, max_size=10))
def test_apikey_rejects_whitespace_only(whitespace):
    result = APIKeyBase.check_api_key(whitespace, auto_error=False)

    assert result is None, (
        f"Whitespace-only API key {repr(whitespace)} should be rejected, "
        f"but got: {repr(result)}"
    )

if __name__ == "__main__":
    test_apikey_rejects_whitespace_only()
```

<details>

<summary>
**Failing input**: `' '`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 15, in <module>
    test_apikey_rejects_whitespace_only()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 6, in test_apikey_rejects_whitespace_only
    def test_apikey_rejects_whitespace_only(whitespace):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 9, in test_apikey_rejects_whitespace_only
    assert result is None, (
           ^^^^^^^^^^^^^^
AssertionError: Whitespace-only API key ' ' should be rejected, but got: ' '
Falsifying example: test_apikey_rejects_whitespace_only(
    whitespace=' ',
)
```
</details>

## Reproducing the Bug

```python
from fastapi.security.api_key import APIKeyBase

# Test whitespace-only strings
result = APIKeyBase.check_api_key("   ", auto_error=False)
print(f"Whitespace key result: {repr(result)}")

result2 = APIKeyBase.check_api_key("\t\r\n", auto_error=False)
print(f"Tab/newline key result: {repr(result2)}")

result3 = APIKeyBase.check_api_key("\x85", auto_error=False)
print(f"Unicode NEL result: {repr(result3)}")

result4 = APIKeyBase.check_api_key("", auto_error=False)
print(f"Empty key result: {repr(result4)}")

# Test with auto_error=True to see if exceptions are raised
print("\nWith auto_error=True:")
try:
    result5 = APIKeyBase.check_api_key("   ", auto_error=True)
    print(f"Whitespace with auto_error=True: {repr(result5)} (no exception raised)")
except Exception as e:
    print(f"Whitespace with auto_error=True: Exception raised - {e}")

try:
    result6 = APIKeyBase.check_api_key("", auto_error=True)
    print(f"Empty string with auto_error=True: {repr(result6)} (no exception raised)")
except Exception as e:
    print(f"Empty string with auto_error=True: Exception raised - {e}")
```

<details>

<summary>
Whitespace strings are accepted while empty strings are rejected
</summary>
```
Whitespace key result: '   '
Tab/newline key result: '\t\r\n'
Unicode NEL result: '\x85'
Empty key result: None

With auto_error=True:
Whitespace with auto_error=True: '   ' (no exception raised)
Empty string with auto_error=True: Exception raised - 403: Not authenticated
```
</details>

## Why This Is A Bug

This bug violates expected authentication behavior by creating an inconsistency in how the `check_api_key` method validates input. The method uses Python's truthiness check (`if not api_key:`) at line 14 of `/home/npc/miniconda/lib/python3.13/site-packages/fastapi/security/api_key.py`, which correctly rejects `None` and empty strings but incorrectly accepts whitespace-only strings.

This inconsistency contradicts security best practices where:
1. Authentication tokens should be meaningful and non-empty
2. Empty strings and whitespace-only strings should be treated equivalently as "no authentication provided"
3. The documentation states that when the key is "not available" it should return None or raise an error, but doesn't clarify that whitespace is considered "available"

The bug affects all three API key authentication classes (`APIKeyQuery`, `APIKeyHeader`, `APIKeyCookie`) since they all inherit from `APIKeyBase` and use the same flawed validation method. When `auto_error=True`, the method should raise an HTTP 403 exception for whitespace-only keys just as it does for empty strings, but instead it returns the whitespace string as valid.

## Relevant Context

The bug is located in the FastAPI security module (version 0.115.12) and affects the core authentication mechanism. The `check_api_key` static method is called by:
- `APIKeyQuery.__call__()` at line 112 for query parameter authentication
- `APIKeyHeader.__call__()` at line 200 for header authentication
- `APIKeyCookie.__call__()` at line 288 for cookie authentication

The documentation for these classes mentions that the dependency result will be "a string containing the key value" and that when `auto_error=False` and the key is "not available", the result will be `None`. However, it doesn't explicitly define what constitutes an "available" vs "not available" key, leading to this ambiguous behavior with whitespace.

FastAPI documentation: https://fastapi.tiangolo.com/tutorial/security/api-key/
Source code: https://github.com/tiangolo/fastapi/blob/master/fastapi/security/api_key.py

## Proposed Fix

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