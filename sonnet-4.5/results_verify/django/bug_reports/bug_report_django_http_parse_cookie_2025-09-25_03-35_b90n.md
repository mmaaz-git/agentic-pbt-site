# Bug Report: django.http.parse_cookie Quoted Cookie Values

**Target**: `django.http.parse_cookie`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `parse_cookie` function incorrectly handles quoted cookie values containing semicolons. It splits on semicolons before unquoting, causing quoted values with embedded semicolons to be parsed incorrectly. This violates RFC 6265, which allows quoted cookie values to contain semicolons.

## Property-Based Test

```python
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
```

**Failing input**: `cookie_dict={'0': ';'}` (or any value containing semicolons)

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

cookie_string = 'session="abc;123"; user=john'
result = parse_cookie(cookie_string)

print(f"Input:    {cookie_string!r}")
print(f"Result:   {result}")
print(f"Expected: {{'session': 'abc;123', 'user': 'john'}}")

assert result == {'session': 'abc;123', 'user': 'john'}
```

**Output:**
```
Input:    'session="abc;123"; user=john'
Result:   {'session': '"abc', '': '123"', 'user': 'john'}
Expected: {'session': 'abc;123', 'user': 'john'}
AssertionError
```

## Why This Is A Bug

1. **RFC 6265 compliance**: Cookie values can be quoted to contain special characters including semicolons. The format is `name="value;with;semicolons"`.

2. **Inconsistent behavior**: The function calls `cookies._unquote(val)`, indicating it intends to handle quoted values, but splits on `;` before unquoting.

3. **Stdlib handles it correctly**: Python's `http.cookies.SimpleCookie.load()` correctly parses the same cookie string:
   ```python
   from http.cookies import SimpleCookie
   sc = SimpleCookie()
   sc.load('session="abc;123"')
   print(sc['session'].value)  # Output: abc;123
   ```

4. **Real-world impact**: Applications that use quoted cookie values with semicolons will have their cookies incorrectly parsed, potentially causing security issues or data loss.

## Fix

The function should parse quoted values before splitting on semicolons, or use a proper cookie parser that respects quoting. Here's a potential fix:

```diff
--- a/django/http/cookie.py
+++ b/django/http/cookie.py
@@ -1,17 +1,11 @@
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
+    # Use SimpleCookie for proper RFC 6265 parsing
+    simple_cookie = cookies.SimpleCookie()
+    try:
+        simple_cookie.load(cookie)
+        return {key: morsel.value for key, morsel in simple_cookie.items()}
+    except cookies.CookieError:
+        # Fall back to empty dict on parse error
+        return {}
```

Alternatively, a more conservative fix would be to implement proper quoted-string parsing according to RFC 6265 Section 4.1.1 before splitting on semicolons.
