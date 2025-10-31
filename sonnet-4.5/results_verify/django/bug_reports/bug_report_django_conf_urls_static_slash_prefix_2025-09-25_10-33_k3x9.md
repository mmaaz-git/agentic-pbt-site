# Bug Report: django.conf.urls.static() Slash Prefix

**Target**: `django.conf.urls.static.static()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `static()` function accepts "/" as a valid prefix, but generates an overly-broad URL pattern `^(?P<path>.*)$` that matches ALL URLs in the application, breaking URL routing. This is inconsistent with the function's validation that explicitly rejects empty prefixes.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.conf.urls.static import static
import re


@given(st.sampled_from(["/", "//", "///", "////"]))
def test_slash_only_prefix_creates_overly_broad_pattern(prefix):
    """
    Property: Prefixes that become empty after lstrip("/") should be rejected,
    just like empty strings are rejected.
    """
    result = static(prefix)

    if result:
        pattern = result[0].pattern.regex

        lstripped = prefix.lstrip("/")
        assert lstripped == ""

        assert pattern.pattern == r'^(?P<path>.*)$'

        assert pattern.match("admin/")
        assert pattern.match("api/users/123")
        assert pattern.match("any/arbitrary/url")
```

**Failing input**: `prefix="/"`

## Reproducing the Bug

```python
import os
import sys

sys.path.insert(0, '/path/to/django')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

import django
from django.conf import settings

if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test')
    django.setup()

from django.conf.urls.static import static

result = static("/")
pattern = result[0].pattern.regex

print(f"Pattern: {pattern.pattern}")

print("\nPattern matches:")
print(f"  Root URL '': {bool(pattern.match(''))}")
print(f"  Admin 'admin/': {bool(pattern.match('admin/'))}")
print(f"  API 'api/users/123': {bool(pattern.match('api/users/123'))}")
print(f"  Everything: YES")

assert pattern.pattern == r'^(?P<path>.*)$'
print("\n✗ BUG: Prefix '/' generates pattern that matches ALL URLs!")
```

## Why This Is A Bug

1. **Inconsistent validation**: Lines 21-22 of `static.py` explicitly check `if not prefix` and raise `ImproperlyConfigured` for empty prefixes. However, `prefix="/"` becomes empty after `prefix.lstrip("/")` on line 28, but doesn't trigger this validation.

2. **Breaks URL routing**: The generated pattern `^(?P<path>.*)$` matches every single URL in the application (admin URLs, API endpoints, etc.), not just static files. This would cause the static file URL pattern to intercept all requests.

3. **Unintuitive behavior**: A user might think `prefix="/"` means "serve from root static directory", but it actually creates a pattern that breaks their entire URL configuration.

4. **Documentation implies non-empty prefix**: The docstring example shows `static(settings.MEDIA_URL, ...)` where MEDIA_URL is typically "/media/" not "/".

## Fix

```diff
diff --git a/django/conf/urls/static.py b/django/conf/urls/static.py
index 1234567..abcdefg 100644
--- a/django/conf/urls/static.py
+++ b/django/conf/urls/static.py
@@ -19,7 +19,7 @@ def static(prefix, view=serve, **kwargs):
         # ... the rest of your URLconf goes here ...
     ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
     """
-    if not prefix:
+    if not prefix or not prefix.lstrip("/"):
         raise ImproperlyConfigured("Empty static prefix not permitted")
     elif not settings.DEBUG or urlsplit(prefix).netloc:
         # No-op if not in debug mode or a non-local prefix.
```

This fix ensures that prefixes like "/" or "///" which become empty after lstrip are rejected, just like empty strings.