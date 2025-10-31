# Bug Report: django.conf.urls.static Catch-All Pattern with Slash-Only Prefix

**Target**: `django.conf.urls.static.static()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `static()` function creates an overly permissive catch-all URL pattern when given a prefix consisting only of slashes (e.g., `"/"`, `"//"`, `"///"`). This happens because `prefix.lstrip("/")` removes all leading slashes, resulting in an empty string that produces the regex `^(?P<path>.*)$`, which matches all URLs.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import re
from unittest.mock import patch
from hypothesis import given, strategies as st, settings
from django.conf.urls.static import static


@given(st.text(alphabet='/', min_size=1, max_size=10))
@settings(max_examples=100)
def test_all_slash_prefix_should_not_create_catchall(prefix):
    """
    Property: A prefix consisting only of slashes should either:
    1. Raise an error (like empty string does), OR
    2. Create a pattern that doesn't match unrelated URLs
    """
    with patch('django.conf.urls.static.settings') as mock_settings:
        mock_settings.DEBUG = True
        result = static(prefix)

        if result:
            pattern_regex = result[0].pattern._regex
            stripped = prefix.lstrip('/')
            if stripped == "":
                assert False, (
                    f"BUG: prefix {repr(prefix)} creates catch-all pattern. "
                    f"After lstrip('/'), prefix becomes empty, creating '^(?P<path>.*)$'"
                )
```

**Failing input**: `"/"`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import re
from unittest.mock import patch
from django.conf.urls.static import static

with patch('django.conf.urls.static.settings') as mock_settings:
    mock_settings.DEBUG = True
    result = static("/")

    pattern_obj = result[0]
    regex = pattern_obj.pattern._regex

    print(f"Pattern regex: {regex}")

    test_urls = ["foo", "bar/baz", "media/image.png", "admin/login"]
    for url in test_urls:
        match = re.match(regex, url)
        if match:
            print(f"'{url}' matches (captured: {match.group('path')})")
```

Output:
```
Pattern regex: ^(?P<path>.*)$
'foo' matches (captured: foo)
'bar/baz' matches (captured: bar/baz)
'media/image.png' matches (captured: media/image.png)
'admin/login' matches (captured: admin/login)
```

## Why This Is A Bug

The `static()` function is intended to create URL patterns for serving static files from a specific prefix directory. When `prefix="/"` is provided:

1. The function does not raise `ImproperlyConfigured` (unlike empty string `""`)
2. `prefix.lstrip("/")` transforms `"/"` to `""` (empty string)
3. The resulting regex `^(?P<path>.*)$` matches **all URLs**, not just those under a specific prefix
4. This would cause all URL requests to be routed to the static file handler in DEBUG mode, breaking the entire application routing

The bug also affects any prefix consisting only of slashes: `"//"`, `"///"`, etc.

## Fix

```diff
--- a/django/conf/urls/static.py
+++ b/django/conf/urls/static.py
@@ -19,7 +19,8 @@ def static(prefix, view=serve, **kwargs):
         # ... the rest of your URLconf goes here ...
     ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
     """
-    if not prefix:
+    stripped_prefix = prefix.lstrip("/") if prefix else ""
+    if not prefix or not stripped_prefix:
         raise ImproperlyConfigured("Empty static prefix not permitted")
     elif not settings.DEBUG or urlsplit(prefix).netloc:
         # No-op if not in debug mode or a non-local prefix.
@@ -27,7 +28,7 @@ def static(prefix, view=serve, **kwargs):
     return [
         re_path(
-            r"^%s(?P<path>.*)$" % re.escape(prefix.lstrip("/")), view, kwargs=kwargs
+            r"^%s(?P<path>.*)$" % re.escape(stripped_prefix), view, kwargs=kwargs
         ),
     ]
```

This fix ensures that prefixes which become empty after stripping leading slashes (like `"/"`, `"//"`) are properly rejected with an `ImproperlyConfigured` exception, just like explicitly empty prefixes.