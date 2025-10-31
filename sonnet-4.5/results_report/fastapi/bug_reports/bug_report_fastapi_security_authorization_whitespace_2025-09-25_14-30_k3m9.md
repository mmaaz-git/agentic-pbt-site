# Bug Report: fastapi.security Authorization Header Leading Whitespace

**Target**: `fastapi.security.utils.get_authorization_scheme_param`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_authorization_scheme_param` function fails to parse Authorization headers with leading whitespace, causing valid HTTP authentication requests to be rejected. According to RFC 7230 Section 3.2.6, servers SHOULD ignore leading and trailing whitespace in header field values.

## Property-Based Test

```python
import pytest
from hypothesis import given, strategies as st, assume
from fastapi.security.utils import get_authorization_scheme_param


@given(
    st.text(min_size=0, max_size=5, alphabet=" \t"),
    st.text(min_size=1).filter(lambda s: not s[0].isspace() and " " not in s and "\t" not in s),
    st.text()
)
def test_leading_whitespace_should_be_ignored(leading_ws, scheme, credentials):
    assume(credentials.strip() == credentials or credentials == "")

    authorization_header = f"{leading_ws}{scheme} {credentials}"

    parsed_scheme, parsed_credentials = get_authorization_scheme_param(authorization_header)

    expected_scheme = scheme
    expected_credentials = credentials

    if leading_ws:
        assert parsed_scheme != expected_scheme or parsed_credentials != expected_credentials, \
            f"Bug: Leading whitespace {leading_ws!r} should be stripped but isn't"
```

**Failing input**: `authorization_header=" Bearer token123"` (single leading space)

## Reproducing the Bug

```python
from fastapi import FastAPI, Depends
from fastapi.security import HTTPBearer
from fastapi.testclient import TestClient

app = FastAPI()
security = HTTPBearer()

@app.get("/protected")
def protected_route(credentials = Depends(security)):
    return {"credentials": credentials.credentials}

client = TestClient(app)

response = client.get("/protected", headers={"Authorization": " Bearer validtoken"})
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
```

Expected output: Status 200 with credentials
Actual output: Status 403 with `{'detail': 'Not authenticated'}`

## Why This Is A Bug

RFC 7230 Section 3.2.6 states: "A recipient SHOULD ignore leading and trailing whitespace in a field value."

The `get_authorization_scheme_param` function uses `str.partition(" ")` which splits on the first space character. When the Authorization header has leading whitespace, the partition splits at that leading space, resulting in an empty scheme and the entire header value (including "Bearer token123") as the credentials.

Example:
- Input: `" Bearer token123"`
- `partition(" ")` returns: `("", " ", "Bearer token123")`
- Result: `scheme=""`, `credentials="Bearer token123"`
- Expected: `scheme="Bearer"`, `credentials="token123"`

This causes authentication to fail because the code checks `if not (authorization and scheme and credentials)`, which evaluates to `True` when scheme is empty.

## Fix

```diff
--- a/fastapi/security/utils.py
+++ b/fastapi/security/utils.py
@@ -1,7 +1,7 @@
 def get_authorization_scheme_param(
     authorization_header_value: Optional[str],
 ) -> Tuple[str, str]:
-    if not authorization_header_value:
+    authorization_header_value = authorization_header_value.strip() if authorization_header_value else None
+    if not authorization_header_value:
         return "", ""
     scheme, _, param = authorization_header_value.partition(" ")
     return scheme, param
```

Alternatively, strip the value after receiving it from the header:

```diff
--- a/fastapi/security/utils.py
+++ b/fastapi/security/utils.py
@@ -1,6 +1,10 @@
 def get_authorization_scheme_param(
     authorization_header_value: Optional[str],
 ) -> Tuple[str, str]:
     if not authorization_header_value:
         return "", ""
+    authorization_header_value = authorization_header_value.strip()
+    if not authorization_header_value:
+        return "", ""
     scheme, _, param = authorization_header_value.partition(" ")
     return scheme, param
```