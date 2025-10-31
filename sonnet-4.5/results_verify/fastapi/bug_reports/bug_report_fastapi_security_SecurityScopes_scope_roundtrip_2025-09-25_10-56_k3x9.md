# Bug Report: SecurityScopes Scope String Round-Trip Failure

**Target**: `fastapi.security.oauth2.SecurityScopes`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`SecurityScopes.scope_str` uses `' '.join()` to create a space-separated string of scopes, but if the scopes are later parsed using `.split()` (as done in `OAuth2PasswordRequestForm`), scopes containing Unicode whitespace characters are lost due to the inconsistency between `.split()` (which splits on ALL whitespace) and `' '.join()` (which only uses spaces).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from fastapi.security.oauth2 import SecurityScopes


@given(st.lists(st.text(alphabet=st.characters(blacklist_characters=" \t\n\r"), min_size=1), min_size=1))
@settings(max_examples=100)
def test_scopes_roundtrip(scopes_list):
    security_scopes = SecurityScopes(scopes=scopes_list)

    assert security_scopes.scopes == scopes_list

    reconstructed_scopes = security_scopes.scope_str.split()
    assert reconstructed_scopes == scopes_list, \
        f"Failed roundtrip: {scopes_list} -> '{security_scopes.scope_str}' -> {reconstructed_scopes}"
```

**Failing input**: `scopes_list=['\x85']`

## Reproducing the Bug

```python
from fastapi.security.oauth2 import SecurityScopes

scopes_list = ['\x85']
security_scopes = SecurityScopes(scopes=scopes_list)

print(f"Input scopes: {scopes_list}")
print(f"scope_str: {security_scopes.scope_str!r}")
print(f"Reconstructed: {security_scopes.scope_str.split()}")

assert security_scopes.scope_str.split() == scopes_list
```

Output:
```
Input scopes: ['\x85']
scope_str: '\x85'
Reconstructed: []
AssertionError
```

## Why This Is A Bug

`SecurityScopes` creates `scope_str` by joining scopes with spaces (line 653 in oauth2.py):
```python
self.scope_str = " ".join(self.scopes)
```

However, `OAuth2PasswordRequestForm` parses scope strings using `.split()` without arguments (line 147):
```python
self.scopes = scope.split()
```

The inconsistency arises because:
- `' '.join()` only uses space characters (`' '`, U+0020) as separators
- `.split()` without arguments splits on ALL Unicode whitespace (space, tab, newline, and many others like `'\x85'` NEL, `'\xa0'` NBSP, etc.)

When a scope contains Unicode whitespace:
1. `' '.join(['\x85'])` produces `'\x85'`
2. `'\x85'.split()` produces `[]` (empty list)
3. The scope is completely lost

This violates the expected round-trip property and causes silent data corruption.

## Fix

The fix is to make the split/join operations consistent. Use `' '.split(' ')` to split only on regular spaces:

```diff
--- a/fastapi/security/oauth2.py
+++ b/fastapi/security/oauth2.py
@@ -144,7 +144,7 @@ class OAuth2PasswordRequestForm:
         self.grant_type = grant_type
         self.username = username
         self.password = password
-        self.scopes = scope.split()
+        self.scopes = scope.split(' ') if scope else []
         self.client_id = client_id
         self.client_secret = client_secret
```

This ensures that:
- `' '.join(scopes).split(' ')` is a true round-trip
- Only regular space characters (U+0020) are used as delimiters
- Scopes can contain other Unicode whitespace without being lost

Note: According to OAuth2 RFC 6749, scope names should not contain spaces, but FastAPI doesn't validate this. The fix ensures consistent behavior even for edge cases.