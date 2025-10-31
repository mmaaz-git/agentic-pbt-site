# Bug Report: fastapi.security APIKey Whitespace Validation

**Target**: `fastapi.security.api_key.APIKeyBase.check_api_key`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `check_api_key` method accepts whitespace-only strings as valid API keys, creating a security vulnerability where authentication can succeed with keys like `"   "`, `"\t"`, `"\r"`, or any of 29 Unicode whitespace characters.

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
```

**Failing input**: `'\r'` (and any whitespace-only string including all 29 Unicode whitespace characters)

## Reproducing the Bug

```python
from fastapi.security.api_key import APIKeyBase

result = APIKeyBase.check_api_key("   ", auto_error=False)
print(f"Whitespace key result: {repr(result)}")

result2 = APIKeyBase.check_api_key("\t\r\n", auto_error=False)
print(f"Tab/newline key result: {repr(result2)}")

result3 = APIKeyBase.check_api_key("\x85", auto_error=False)
print(f"Unicode NEL result: {repr(result3)}")

result4 = APIKeyBase.check_api_key("", auto_error=False)
print(f"Empty key result: {repr(result4)}")
```

Expected output: All should return `None`

Actual output:
```
Whitespace key result: '   '
Tab/newline key result: '\t\r\n'
Unicode NEL result: '\x85'
Empty key result: None
```

## Why This Is A Bug

The `check_api_key` method uses Python's truthiness check `if not api_key:` which treats empty strings as falsy but whitespace strings as truthy. This creates inconsistent and insecure behavior:

1. Empty string `""` is correctly rejected
2. Whitespace-only strings (29 Unicode whitespace characters) are incorrectly accepted

This security vulnerability could allow attackers to bypass API key authentication by sending whitespace-only values, particularly in scenarios where:
- The application doesn't perform additional validation after `check_api_key`
- Middleware or proxies normalize or trim headers inconsistently
- Log analysis might miss whitespace-only authentication attempts
- The whitespace key doesn't match any stored key but still passes initial validation

The bug affects:
- `APIKeyQuery` - query parameter authentication
- `APIKeyHeader` - header authentication
- `APIKeyCookie` - cookie authentication

All three classes use the flawed `check_api_key` method.

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

This fix ensures that both empty strings and whitespace-only strings (including all Unicode whitespace) are consistently rejected.