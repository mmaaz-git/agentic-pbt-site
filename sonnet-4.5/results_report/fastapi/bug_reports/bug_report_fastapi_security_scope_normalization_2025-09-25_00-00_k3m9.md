# Bug Report: fastapi.security Scope String Normalization Inconsistency

**Target**: `fastapi.security.OAuth2PasswordRequestForm` and `fastapi.security.SecurityScopes`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `OAuth2PasswordRequestForm` and `SecurityScopes` classes have an asymmetric implementation of scope string handling: they use `" ".join()` to create scope strings but `split()` (without arguments) to parse them. This causes silent normalization of whitespace, violating the round-trip property and potentially causing unexpected behavior with user input.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi.security import OAuth2PasswordRequestForm, SecurityScopes

@given(st.lists(st.text(min_size=1).filter(lambda x: not x.isspace() and " " not in x)))
def test_oauth2_scope_round_trip(scope_list):
    scope_string = " ".join(scope_list)
    form = OAuth2PasswordRequestForm(
        username="test_user",
        password="test_pass",
        scope=scope_string
    )
    assert form.scopes == scope_list
    reconstructed = " ".join(form.scopes)
    assert reconstructed == scope_string
```

**Failing input**: `scope_list=['0\r']` - scope with trailing carriage return gets stripped

## Reproducing the Bug

```python
from fastapi.security import OAuth2PasswordRequestForm, SecurityScopes

scope_string = "read  write"
form = OAuth2PasswordRequestForm(
    username="test",
    password="test",
    scope=scope_string
)
print(f"Input:  {scope_string!r}")
print(f"Output: {' '.join(form.scopes)!r}")

security_scopes = SecurityScopes(scopes=['\r'])
print(f"Input scopes:  {['\r']!r}")
print(f"After split:   {security_scopes.scope_str.split()!r}")
```

**Output:**
```
Input:  'read  write'
Output: 'read write'
Input scopes:  ['\r']
After split:   []
```

## Why This Is A Bug

1. **Asymmetric operations**: Uses `" ".join()` (space-only) but `split()` (all whitespace including `\t`, `\n`, `\r`)
2. **Silent normalization**: Multiple spaces between scopes get collapsed to single spaces
3. **Data loss**: Whitespace-only strings or trailing whitespace in scopes are silently removed
4. **Violates OAuth2 spec**: OAuth2 scopes should be opaque strings separated by spaces, not normalized

## Fix

```diff
--- a/fastapi/security/oauth2.py
+++ b/fastapi/security/oauth2.py
@@ -336,7 +336,7 @@ class OAuth2PasswordRequestForm:
         self.grant_type = grant_type
         self.username = username
         self.password = password
-        self.scopes = scope.split()
+        self.scopes = scope.split(" ") if scope else []
         self.client_id = client_id
         self.client_secret = client_secret
```

Note: This fix makes `split()` use space as delimiter (matching `join(" ")`), but it may break existing code that relies on the normalization behavior. An alternative would be to document the normalization behavior explicitly.