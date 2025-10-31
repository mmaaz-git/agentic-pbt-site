# Bug Report: django.conf.urls.static Creates Catch-All Pattern with Slash-Only Prefix

**Target**: `django.conf.urls.static.static()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `static()` function creates a catch-all URL pattern that matches every URL in the application when given a prefix consisting only of slashes (e.g., "/"), breaking Django's entire URL routing in DEBUG mode.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

import re
from unittest.mock import patch
from hypothesis import given, strategies as st, settings
import django
from django.conf import settings as django_settings

# Configure Django settings
django_settings.configure(DEBUG=True)

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

# Run the test
test_all_slash_prefix_should_not_create_catchall()
```

<details>

<summary>
**Failing input**: `'/'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 38, in <module>
    test_all_slash_prefix_should_not_create_catchall()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 17, in test_all_slash_prefix_should_not_create_catchall
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 32, in test_all_slash_prefix_should_not_create_catchall
    assert False, (
           ^^^^^
AssertionError: BUG: prefix '/' creates catch-all pattern. After lstrip('/'), prefix becomes empty, creating '^(?P<path>.*)$'
Falsifying example: test_all_slash_prefix_should_not_create_catchall(
    prefix='/',
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

import re
from unittest.mock import patch
import django
from django.conf import settings

# Configure Django settings
settings.configure(DEBUG=True)

from django.conf.urls.static import static

# Test the bug with "/" prefix
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

<details>

<summary>
Demonstrates catch-all pattern matching all URLs
</summary>
```
Pattern regex: ^(?P<path>.*)$
'foo' matches (captured: foo)
'bar/baz' matches (captured: bar/baz)
'media/image.png' matches (captured: media/image.png)
'admin/login' matches (captured: admin/login)
```
</details>

## Why This Is A Bug

The `static()` function is designed to serve static files from a specific prefix directory during development (DEBUG mode). However, when a prefix consisting only of slashes like "/" is provided, the function's logic fails catastrophically:

1. The function checks `if not prefix:` at line 21, which evaluates to False for "/" (a non-empty string)
2. At line 28, the function calls `prefix.lstrip("/")` which transforms "/" into "" (empty string)
3. This empty string is passed to `re.escape()`, resulting in the regex pattern `^(?P<path>.*)$`
4. This regex matches ANY URL path, not just URLs under a specific prefix
5. In DEBUG mode, this would cause ALL URL requests in the entire Django application to be incorrectly routed to the static file handler

This violates the documented behavior that static() should "Return a URL pattern for serving files" from a specific prefix. According to Django's URL routing documentation, URL patterns should match specific path prefixes, not capture all possible URLs. The function already has logic to reject empty prefixes with `ImproperlyConfigured`, but this check fails to catch prefixes that become empty after stripping slashes.

## Relevant Context

The bug affects any prefix consisting entirely of forward slashes: "/", "//", "///", etc. All of these get stripped to an empty string by `lstrip("/")`.

The `static()` function is commonly used in Django projects' URL configuration during development:
- Django documentation: https://docs.djangoproject.com/en/stable/howto/static-files/#serving-files-uploaded-by-a-user-during-development
- Source code location: `/django/conf/urls/static.py:10-30`

This is particularly dangerous because:
- It silently breaks the entire application's routing without any error message
- Developers might attempt to use "/" thinking it would serve files from the root
- The resulting catch-all pattern would intercept ALL URLs before they reach the actual application views

## Proposed Fix

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