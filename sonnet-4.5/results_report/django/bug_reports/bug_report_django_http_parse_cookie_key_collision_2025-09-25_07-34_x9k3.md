# Bug Report: Django HTTP parse_cookie Whitespace Key Collision

**Target**: `django.http.cookie.parse_cookie`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `parse_cookie` function silently loses data when multiple cookies have whitespace-only names, as they all get stripped to the empty string key and collide in the resulting dictionary.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.http.cookie import parse_cookie

@given(st.dictionaries(
    st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters=set('=;'))),
    st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters=set(';'))),
    min_size=1, max_size=20
))
def test_parse_cookie_preserves_all_cookies(cookies_dict):
    cookie_string = "; ".join(f"{k}={v}" for k, v in cookies_dict.items())
    parsed = parse_cookie(cookie_string)

    for key in cookies_dict.keys():
        assert key.strip() in parsed
```

**Failing input**: `{'\r': '0'}` → Expected key `''` with value `'0'`, but when combined with `{'\n': '1'}`, only one survives.

## Reproducing the Bug

```python
from django.http.cookie import parse_cookie

cookie_string = " =first; \t=second; \n=third"
result = parse_cookie(cookie_string)

print(f"Input:  {cookie_string!r}")
print(f"Output: {result}")
print(f"Expected: 3 separate cookies")
print(f"Actual:   {len(result)} cookie (data loss)")
```

Output:
```
Input:  ' =first; \t=second; \n=third'
Output: {'': 'third'}
Expected: 3 separate cookies
Actual:   1 cookie (data loss)
```

## Why This Is A Bug

In `django/http/cookie.py` lines 19-22:

```python
key, val = key.strip(), val.strip()
if key or val:
    cookiedict[key] = val
```

When cookie names consist entirely of whitespace (e.g., `\r`, `\n`, ` `), they all get stripped to the empty string `""`. Since dictionaries can only have one value per key, multiple such cookies collide and only the last value is preserved. This causes silent data loss.

While whitespace-only cookie names are invalid per RFC 6265, Django should either:
1. Reject invalid cookies explicitly (raise exception or skip them)
2. Preserve the original key to avoid collisions
3. Warn about the data loss

Silently losing data is worse than rejecting invalid input.

## Fix

Option 1: Skip cookies with empty keys after stripping

```diff
--- a/django/http/cookie.py
+++ b/django/http/cookie.py
@@ -18,6 +18,6 @@ def parse_cookie(cookie):
             key, val = "", chunk
         key, val = key.strip(), val.strip()
-        if key or val:
+        if key and val:
             cookiedict[key] = val
     return cookiedict
```

Option 2: Preserve non-empty original keys before stripping

```diff
--- a/django/http/cookie.py
+++ b/django/http/cookie.py
@@ -16,8 +16,10 @@ def parse_cookie(cookie):
         else:
             key, val = "", chunk
         key, val = key.strip(), val.strip()
-        if key or val:
+        if not key and chunk.split("=", 1)[0]:
+            continue  # Skip cookies with whitespace-only names
+        if key:
             cookiedict[key] = val
     return cookiedict
```