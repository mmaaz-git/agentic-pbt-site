# Bug Report: django.conf.urls.static Slash-Only Prefix Creates Catch-All Pattern

**Target**: `django.conf.urls.static.static()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `static()` function accepts a slash-only prefix (e.g., "/", "//", "///") which, after `lstrip("/")`, becomes an empty string. This creates a catch-all regex pattern `^(?P<path>.*)$` that matches all URLs, causing incorrect routing behavior.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings as hyp_settings
from django.conf import settings
from django.conf.urls.static import static

settings.configure(DEBUG=True)

@hyp_settings(max_examples=200)
@given(st.integers(min_value=1, max_value=100))
def test_static_slash_only_prefix_bug(num_slashes):
    prefix = '/' * num_slashes
    result = static(prefix)

    if result:
        pattern_obj = result[0].pattern
        regex_pattern = pattern_obj.regex.pattern

        stripped = prefix.lstrip('/')

        if not stripped:
            assert regex_pattern != r'^(?P<path>.*)$', \
                f"BUG: Slash-only prefix '{prefix}' creates catch-all regex: {regex_pattern}"
```

**Failing input**: `num_slashes=1` (or any positive integer)

## Reproducing the Bug

```python
from django.conf import settings
from django.conf.urls.static import static

settings.configure(DEBUG=True)

prefix = "/"
result = static(prefix)
pattern = result[0].pattern.regex

print(f"Prefix: '{prefix}'")
print(f"Regex pattern: {pattern.pattern}")

test_paths = ["media/file.jpg", "admin/login", "api/users", ""]
for path in test_paths:
    matches = pattern.match(path) is not None
    print(f"  '{path}' matches: {matches}")
```

Output:
```
Prefix: '/'
Regex pattern: ^(?P<path>.*)$
  'media/file.jpg' matches: True
  'admin/login' matches: True
  'api/users' matches: True
  '' matches: True
```

## Why This Is A Bug

1. **Incorrect validation**: The code checks `if not prefix:` but "/" is truthy, so it passes validation
2. **Unintended behavior**: After `prefix.lstrip("/")` on "/", the result is an empty string
3. **Catch-all pattern**: `re.escape("")` produces "" which creates the pattern `^(?P<path>.*)$`
4. **Routing chaos**: This pattern matches **every** URL, causing static file handling to intercept all requests
5. **Silent failure**: No error is raised, making this a silent logic bug

The function is designed to serve static files under a specific prefix, but a slash-only prefix defeats this purpose entirely.

## Fix

Add validation to reject prefixes that become empty after stripping leading slashes:

```diff
--- a/django/conf/urls/static.py
+++ b/django/conf/urls/static.py
@@ -20,7 +20,8 @@ def static(prefix, view=serve, **kwargs):
     """
     if not prefix:
         raise ImproperlyConfigured("Empty static prefix not permitted")
-    elif not settings.DEBUG or urlsplit(prefix).netloc:
+    stripped_prefix = prefix.lstrip("/")
+    if not stripped_prefix or not settings.DEBUG or urlsplit(prefix).netloc:
         # No-op if not in debug mode or a non-local prefix.
         return []
     return [
```

Alternatively, check before lstrip:

```diff
--- a/django/conf/urls/static.py
+++ b/django/conf/urls/static.py
@@ -20,6 +20,8 @@ def static(prefix, view=serve, **kwargs):
     """
     if not prefix:
         raise ImproperlyConfigured("Empty static prefix not permitted")
+    if prefix.lstrip("/") == "":
+        raise ImproperlyConfigured("Prefix cannot consist only of slashes")
     elif not settings.DEBUG or urlsplit(prefix).netloc:
         # No-op if not in debug mode or a non-local prefix.
         return []
```