# Bug Report: django.conf.urls.static Whitespace-Only Prefix

**Target**: `django.conf.urls.static.static`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `static()` function accepts whitespace-only prefixes (e.g., `' '`, `'\t'`, `'\n'`) when it should reject them as empty. The validation check `if not prefix:` only catches truly empty strings but allows whitespace-only strings, leading to malformed URL patterns.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from django.test import override_settings
from django.conf.urls.static import static
from django.core.exceptions import ImproperlyConfigured
import pytest

@given(st.text(alphabet=' \t\n', min_size=1, max_size=10))
@settings(max_examples=50)
def test_static_whitespace_prefix_should_raise(prefix):
    with override_settings(DEBUG=True):
        with pytest.raises(ImproperlyConfigured):
            static(prefix)
```

**Failing input**: `' '` (single space, or any whitespace-only string)

## Reproducing the Bug

```python
import django
from django.conf import settings as django_settings

if not django_settings.configured:
    django_settings.configure(DEBUG=True, SECRET_KEY='test')
    django.setup()

from django.conf.urls.static import static
from django.core.exceptions import ImproperlyConfigured

result = static(' ')
print(result)
```

## Why This Is A Bug

1. The error message states "Empty static prefix not permitted", indicating that empty/whitespace-only prefixes should be rejected
2. Whitespace-only prefixes are semantically empty and serve no valid purpose
3. Accepting whitespace-only prefixes creates malformed URL patterns that could cause routing issues
4. The behavior is inconsistent - truly empty strings are rejected but whitespace-only strings are not

## Fix

```diff
--- a/django/conf/urls/static.py
+++ b/django/conf/urls/static.py
@@ -18,7 +18,7 @@ def static(prefix, view=serve, **kwargs):
         # ... the rest of your URLconf goes here ...
     ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
     """
-    if not prefix:
+    if not prefix or not prefix.strip():
         raise ImproperlyConfigured("Empty static prefix not permitted")
     elif not settings.DEBUG or urlsplit(prefix).netloc:
         # No-op if not in debug mode or a non-local prefix.
```