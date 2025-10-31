# Bug Report: fastapi.security.SecurityScopes Invalid Scope Handling

**Target**: `fastapi.security.SecurityScopes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `SecurityScopes` class accepts whitespace-only or empty strings in its `scopes` parameter but produces an inconsistent `scope_str` that cannot be parsed back to the original scopes list, violating the expected round-trip property for OAuth2 scope handling.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi.security import SecurityScopes

@given(st.lists(st.text(min_size=1)))
def test_security_scopes_scope_str_round_trip(scopes_list):
    scopes = SecurityScopes(scopes=scopes_list)
    reconstructed = scopes.scope_str.split()
    assert reconstructed == scopes.scopes
```

**Failing input**: `scopes_list=[' ']`

## Reproducing the Bug

```python
from fastapi.security import SecurityScopes

scopes_list = [' ']
scopes = SecurityScopes(scopes=scopes_list)

print(f"Original scopes: {scopes.scopes}")
print(f"scope_str: {repr(scopes.scope_str)}")
print(f"Reconstructed: {scopes.scope_str.split()}")

assert scopes.scopes == [' ']
assert scopes.scope_str == ' '
assert scopes.scope_str.split() == []
```

**Output:**
```
Original scopes: [' ']
scope_str: ' '
Reconstructed: []
```

The round-trip property is violated: `[' '] != []`

## Why This Is A Bug

The `SecurityScopes` class is designed to work with OAuth2 scopes, which according to the OAuth2 specification are non-empty, non-whitespace strings. However, the class constructor accepts any `list[str]` without validation:

1. **No input validation**: The class accepts `scopes=[' ']`, `scopes=['']`, or `scopes=['read', ' ', 'write']` without raising an error
2. **Inconsistent state**: The resulting `scope_str` property cannot be correctly parsed back to the original scopes using `.split()`
3. **Violates documented behavior**: The docstring states that `scope_str` contains "scopes separated by spaces, as defined in the OAuth2 specification", but whitespace-only scopes are not valid per OAuth2 spec

While normal FastAPI usage (where scopes come from `OAuth2PasswordRequestForm`) won't trigger this bug (since that class uses `.split()` which filters whitespace), users who manually instantiate `SecurityScopes` may encounter unexpected behavior.

## Fix

Add input validation to reject invalid scopes:

```diff
diff --git a/fastapi/security/oauth2.py b/fastapi/security/oauth2.py
index 1234567..abcdefg 100644
--- a/fastapi/security/oauth2.py
+++ b/fastapi/security/oauth2.py
@@ -625,7 +625,11 @@ class SecurityScopes:

     def __init__(
         self,
-        scopes: Optional[List[str]] = None,
+        scopes: Annotated[
+            Optional[List[str]],
+            Doc(
+                """
+                This will be filled by FastAPI.
+                """
+            ),
+        ] = None,
     ):
-        self.scopes: List[str] = scopes or []
+        validated_scopes = scopes or []
+        if validated_scopes:
+            for scope in validated_scopes:
+                if not scope or not scope.strip():
+                    raise ValueError(
+                        f"Invalid scope: {repr(scope)}. "
+                        "OAuth2 scopes must be non-empty, non-whitespace strings."
+                    )
+        self.scopes: List[str] = validated_scopes
         self.scope_str: str = " ".join(self.scopes)
```

Alternatively, filter out invalid scopes silently:

```diff
-        self.scopes: List[str] = scopes or []
+        self.scopes: List[str] = [s for s in (scopes or []) if s and s.strip()]
```

Or at minimum, document this limitation in the class docstring.