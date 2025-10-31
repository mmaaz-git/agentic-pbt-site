# Bug Report: django.http.parse_cookie Incorrectly Handles Semicolons in Cookie Values

**Target**: `django.http.parse_cookie`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `parse_cookie` function incorrectly parses cookie values containing semicolons by splitting on semicolons before unquoting, causing both bare semicolons and quoted values with embedded semicolons to be parsed incorrectly.

## Property-Based Test

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        DEFAULT_CHARSET='utf-8',
    )
    django.setup()

from django.http import parse_cookie
from hypothesis import given, strategies as st, settings as hypothesis_settings

@given(st.dictionaries(
    st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'),
        min_codepoint=32, max_codepoint=126
    )),
    st.text(max_size=100, alphabet=st.characters(
        min_codepoint=32, max_codepoint=126
    )),
    min_size=1, max_size=10
))
@hypothesis_settings(max_examples=500)
def test_parse_cookie_preserves_data(cookie_dict):
    cookie_string = "; ".join(f"{k}={v}" for k, v in cookie_dict.items())
    parsed = parse_cookie(cookie_string)

    for key, value in cookie_dict.items():
        assert key in parsed, f"Key {key} not in parsed cookies"
        assert parsed[key] == value, f"Value mismatch for {key}: {parsed[key]} vs {value}"

if __name__ == "__main__":
    test_parse_cookie_preserves_data()
```

<details>

<summary>
**Failing input**: `cookie_dict={'0': ';'}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 35, in <module>
    test_parse_cookie_preserves_data()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 16, in test_parse_cookie_preserves_data
    st.text(min_size=1, max_size=50, alphabet=st.characters(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 32, in test_parse_cookie_preserves_data
    assert parsed[key] == value, f"Value mismatch for {key}: {parsed[key]} vs {value}"
           ^^^^^^^^^^^^^^^^^^^^
AssertionError: Value mismatch for 0:  vs ;
Falsifying example: test_parse_cookie_preserves_data(
    cookie_dict={'0': ';'},
)
```
</details>

## Reproducing the Bug

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        DEFAULT_CHARSET='utf-8',
    )
    django.setup()

from django.http import parse_cookie

# Test case 1: Cookie with quoted value containing semicolon
cookie_string = 'session="abc;123"; user=john'
result = parse_cookie(cookie_string)

print(f"Input:    {cookie_string!r}")
print(f"Result:   {result}")
print(f"Expected: {{'session': 'abc;123', 'user': 'john'}}")
print()

# Test case 2: Simple semicolon in value
cookie_string2 = '0=;'
result2 = parse_cookie(cookie_string2)

print(f"Input:    {cookie_string2!r}")
print(f"Result:   {result2}")
print(f"Expected: {{'0': ';'}}")
print()

# Compare with Python's SimpleCookie
from http.cookies import SimpleCookie

print("Python's SimpleCookie behavior:")
sc = SimpleCookie()
sc.load('session="abc;123"; user=john')
print(f"SimpleCookie result: {dict((k, v.value) for k, v in sc.items())}")

# Verify the assertion failure
try:
    assert result == {'session': 'abc;123', 'user': 'john'}
    print("Assertion passed")
except AssertionError:
    print("AssertionError: parse_cookie incorrectly handled quoted semicolons")
```

<details>

<summary>
AssertionError: Cookie values with semicolons are incorrectly parsed
</summary>
```
Input:    'session="abc;123"; user=john'
Result:   {'session': '"abc', '': '123"', 'user': 'john'}
Expected: {'session': 'abc;123', 'user': 'john'}

Input:    '0=;'
Result:   {'0': ''}
Expected: {'0': ';'}

Python's SimpleCookie behavior:
SimpleCookie result: {'session': 'abc;123', 'user': 'john'}
AssertionError: parse_cookie incorrectly handled quoted semicolons
```
</details>

## Why This Is A Bug

The `parse_cookie` function has a fundamental logic error in its implementation order. It splits the entire cookie string on semicolons (`;`) at line 12 before attempting to unquote values at line 22. This breaks both:

1. **Bare semicolons in values**: A cookie like `0=;` gets parsed as `{'0': ''}` instead of `{'0': ';'}` because the semicolon is treated as a separator and discarded.

2. **Quoted values with semicolons**: A cookie like `session="abc;123"` gets incorrectly split into `session="abc` and `123"`, resulting in `{'session': '"abc', '': '123"'}` instead of the correct `{'session': 'abc;123'}`.

The function's use of `cookies._unquote()` at line 22 shows clear intent to handle quoted values, but this happens too late - after the string has already been incorrectly split. This violates the principle that quoted strings should be treated as atomic units during parsing.

## Relevant Context

The bug is located in `/django/http/cookie.py` at lines 7-23. The function is used throughout Django's request handling to parse HTTP Cookie headers in both WSGI and ASGI handlers.

Python's standard library `SimpleCookie` (which Django imports and exposes from the same module) handles these cases correctly:
- Django exposes `SimpleCookie` at line 4 of the same file
- `SimpleCookie` correctly parses `'session="abc;123"'` as `{'session': 'abc;123'}`
- Django's `parse_cookie` incorrectly parses it as `{'session': '"abc', '': '123"'}`

This inconsistency within the same module is problematic, as developers may expect similar behavior from functions in the same module that handle the same data format.

Documentation references:
- Django's parse_cookie source: https://github.com/django/django/blob/main/django/http/cookie.py
- RFC 2109 (allows quoted semicolons): https://www.rfc-editor.org/rfc/rfc2109.html
- RFC 6265 (current standard, doesn't allow semicolons even quoted): https://www.rfc-editor.org/rfc/rfc6265.html

## Proposed Fix

```diff
--- a/django/http/cookie.py
+++ b/django/http/cookie.py
@@ -7,17 +7,11 @@ SimpleCookie = cookies.SimpleCookie
 def parse_cookie(cookie):
     """
     Return a dictionary parsed from a `Cookie:` header string.
     """
-    cookiedict = {}
-    for chunk in cookie.split(";"):
-        if "=" in chunk:
-            key, val = chunk.split("=", 1)
-        else:
-            # Assume an empty name per
-            # https://bugzilla.mozilla.org/show_bug.cgi?id=169091
-            key, val = "", chunk
-        key, val = key.strip(), val.strip()
-        if key or val:
-            # unquote using Python's algorithm.
-            cookiedict[key] = cookies._unquote(val)
-    return cookiedict
+    # Use SimpleCookie for proper RFC-compliant parsing
+    simple_cookie = cookies.SimpleCookie()
+    try:
+        simple_cookie.load(cookie)
+        return {key: morsel.value for key, morsel in simple_cookie.items()}
+    except cookies.CookieError:
+        # Fall back to empty dict on parse error
+        return {}
```