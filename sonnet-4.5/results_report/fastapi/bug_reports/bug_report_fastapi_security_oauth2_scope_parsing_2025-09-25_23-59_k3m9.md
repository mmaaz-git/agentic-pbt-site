# Bug Report: OAuth2PasswordRequestForm Scope Parsing Violates OAuth2 Specification

**Target**: `fastapi.security.OAuth2PasswordRequestForm`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`OAuth2PasswordRequestForm` uses `split()` instead of `split(" ")` to parse the scope parameter, which violates the OAuth2 specification (RFC 6749) and can lead to incorrect scope parsing when non-space whitespace characters are present.

## Property-Based Test

```python
from hypothesis import given, strategies as st


@given(st.lists(st.text(min_size=1).filter(lambda x: " " not in x), min_size=1))
def test_oauth2_scope_spec_compliance(scopes_list):
    scope_string = " ".join(scopes_list)

    parsed_scopes_with_split = scope_string.split()
    parsed_scopes_with_space_split = scope_string.split(" ")

    assert parsed_scopes_with_split == parsed_scopes_with_space_split
```

**Failing input**: `scopes_list = ['\r']`

## Reproducing the Bug

```python
malicious_scope = "read\nwrite"

parsed = malicious_scope.split()
print(f"Parsed scopes: {parsed}")

expected_per_spec = malicious_scope.split(" ")
print(f"Expected per OAuth2 spec: {expected_per_spec}")

assert parsed == ['read', 'write']
assert expected_per_spec == ['read\nwrite']
```

Output:
```
Parsed scopes: ['read', 'write']
Expected per OAuth2 spec: ['read\nwrite']
```

## Why This Is A Bug

According to OAuth2 specification (RFC 6749, Section 3.3), scope values are defined as:

```
scope = scope-token *( SP scope-token )
```

Where `SP` is the space character (0x20). The specification explicitly requires scopes to be separated by **space characters only**, not any whitespace.

The current implementation in `oauth2.py` line 147 uses:

```python
self.scopes = scope.split()
```

Python's `split()` without arguments splits on **any** whitespace character (spaces, tabs, newlines, carriage returns, etc.) and strips leading/trailing whitespace. This means:

1. A scope string `"read\nwrite"` gets incorrectly parsed as two scopes: `['read', 'write']`
2. According to the spec, it should be treated as one (invalid) scope: `['read\nwrite']`

This has potential security implications, as a client could inject additional scopes by using non-space whitespace characters to bypass validation.

## Fix

```diff
--- a/oauth2.py
+++ b/oauth2.py
@@ -144,7 +144,7 @@ class OAuth2PasswordRequestForm:
         self.grant_type = grant_type
         self.username = username
         self.password = password
-        self.scopes = scope.split()
+        self.scopes = [s for s in scope.split(" ") if s]
         self.client_id = client_id
         self.client_secret = client_secret
```

Note: The list comprehension `[s for s in scope.split(" ") if s]` is needed to handle multiple consecutive spaces correctly, filtering out empty strings that would result from `split(" ")` with multiple spaces.