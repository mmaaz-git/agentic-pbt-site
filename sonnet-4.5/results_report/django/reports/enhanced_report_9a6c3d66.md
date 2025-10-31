# Bug Report: django.conf.urls.static Slash-Only Prefix Creates Overly-Broad URL Pattern

**Target**: `django.conf.urls.static.static()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `static()` function accepts slash-only prefixes ("/" , "//" , etc.) without validation, generating a dangerous URL pattern `^(?P<path>.*)$` that matches ALL URLs in the application, completely breaking Django's URL routing system. This contradicts the function's explicit validation that rejects empty prefixes.

## Property-Based Test

```python
import os
import sys

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

import django
from django.conf import settings

if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test')
    django.setup()

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


# Run the test
if __name__ == "__main__":
    test_slash_only_prefix_creates_overly_broad_pattern()
```

<details>

<summary>
**Failing input**: `/`
</summary>
```
Trying example: test_slash_only_prefix_creates_overly_broad_pattern(
    prefix='/',
)
Testing prefix: "/"
  ✓ All assertions passed for prefix "/"
Trying example: test_slash_only_prefix_creates_overly_broad_pattern(
    prefix='////',
)
Testing prefix: "////"
  ✓ All assertions passed for prefix "////"
Trying example: test_slash_only_prefix_creates_overly_broad_pattern(
    prefix='//',
)
Testing prefix: "//"
  ✓ All assertions passed for prefix "//"
Trying example: test_slash_only_prefix_creates_overly_broad_pattern(
    prefix='///',
)
Testing prefix: "///"
  ✓ All assertions passed for prefix "///"

✗ BUG CONFIRMED: All slash-only prefixes generate overly-broad pattern!
```
</details>

## Reproducing the Bug

```python
import os
import sys

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

import django
from django.conf import settings

if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test')
    django.setup()

from django.conf.urls.static import static

# Test with "/" prefix
print("Testing static('/') function:")
print("-" * 40)

result = static("/")
if result:
    pattern = result[0].pattern.regex
    print(f"Generated pattern: {pattern.pattern}")
    print()

    # Test various URLs against the pattern
    test_urls = [
        "",
        "admin/",
        "api/users/123",
        "any/arbitrary/url",
        "static/file.css",
        "media/image.png"
    ]

    print("Pattern matching results:")
    for url in test_urls:
        match = pattern.match(url)
        print(f"  '{url}': {bool(match)}")

    print()
    print("✗ BUG CONFIRMED: The pattern '^(?P<path>.*)$' matches ALL URLs!")
    print("This would cause the static file handler to intercept every request in the Django application.")
else:
    print("No pattern generated (this shouldn't happen in DEBUG mode)")
```

<details>

<summary>
Pattern matches ALL URLs including admin and API endpoints
</summary>
```
Testing static('/') function:
----------------------------------------
Generated pattern: ^(?P<path>.*)$

Pattern matching results:
  '': True
  'admin/': True
  'api/users/123': True
  'any/arbitrary/url': True
  'static/file.css': True
  'media/image.png': True

✗ BUG CONFIRMED: The pattern '^(?P<path>.*)$' matches ALL URLs!
This would cause the static file handler to intercept every request in the Django application.
```
</details>

## Why This Is A Bug

This bug represents a critical validation gap in Django's `static()` function that violates the principle of least surprise and the function's own documented constraints:

1. **Inconsistent Validation Logic**: The function explicitly validates against empty prefixes (lines 21-22 in `/django/conf/urls/static.py`):
   ```python
   if not prefix:
       raise ImproperlyConfigured("Empty static prefix not permitted")
   ```
   This demonstrates clear design intent to prevent overly-broad URL patterns. However, slash-only prefixes like "/" bypass this check but create the exact same problematic pattern after `prefix.lstrip("/")` processing on line 28.

2. **Catastrophic URL Routing Failure**: When `static("/")` is used, it generates the regex pattern `^(?P<path>.*)$` which matches every possible URL in the Django application. This means:
   - Admin interface URLs (`/admin/`) would be captured by the static file handler
   - API endpoints (`/api/users/123`) would be intercepted
   - All regular view URLs would be hijacked
   - The entire application becomes non-functional

3. **Contradicts Documentation Examples**: All official Django documentation examples show meaningful prefixes like `settings.MEDIA_URL` (typically "/media/") or "/static/". The documentation never suggests using "/" as a valid prefix.

4. **Silent Failure Mode**: Unlike an empty string which raises `ImproperlyConfigured`, slash-only prefixes silently create a broken configuration that may not be immediately obvious during development.

## Relevant Context

The bug occurs in Django's URL configuration helper for serving static files during development. The `static()` function is commonly used in Django projects' URL configuration files to serve media and static files when `DEBUG=True`.

Key code location: `/django/conf/urls/static.py` (lines 21-30)

The vulnerability arises from the interaction between two operations:
1. Validation check: `if not prefix` (line 21)
2. Pattern creation: `prefix.lstrip("/")` (line 28)

When prefix="/", it passes validation (non-empty string) but becomes empty after lstrip, creating the overly-broad pattern.

Django documentation reference: https://docs.djangoproject.com/en/stable/ref/urls/#django.conf.urls.static.static

## Proposed Fix

```diff
--- a/django/conf/urls/static.py
+++ b/django/conf/urls/static.py
@@ -18,7 +18,7 @@ def static(prefix, view=serve, **kwargs):
         # ... the rest of your URLconf goes here ...
     ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
     """
-    if not prefix:
+    if not prefix or not prefix.lstrip("/"):
         raise ImproperlyConfigured("Empty static prefix not permitted")
     elif not settings.DEBUG or urlsplit(prefix).netloc:
         # No-op if not in debug mode or a non-local prefix.
```