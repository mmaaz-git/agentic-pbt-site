# Bug Report: django.conf.urls.static Incomplete Prefix Validation

**Target**: `django.conf.urls.static.static()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `static()` function validates that the prefix is not empty, but fails to validate that it doesn't become empty after internal `lstrip('/')` processing. This allows slash-only prefixes like `'/'` to create overly broad URL patterns.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.conf.urls.static import static
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings
import pytest


@given(st.text(alphabet='/', min_size=1, max_size=10))
@override_settings(DEBUG=True)
def test_static_only_slashes(prefix):
    with pytest.raises(ImproperlyConfigured):
        static(prefix)
```

**Failing input**: `'/'` (or any string containing only slashes)

## Reproducing the Bug

```python
from django.conf import settings
from django.conf.urls.static import static
from django.core.exceptions import ImproperlyConfigured

settings.configure(DEBUG=True, SECRET_KEY='test')

result = static('/')
print(f"Pattern: {result[0].pattern.regex.pattern}")

try:
    static('')
except ImproperlyConfigured as e:
    print(f"Empty string raises: {e}")
```

**Output:**
```
Pattern: ^(?P<path>.*)$
Empty string raises: Empty static prefix not permitted
```

## Why This Is A Bug

The validation check at line 21 of `static.py` is:
```python
if not prefix:
    raise ImproperlyConfigured("Empty static prefix not permitted")
```

However, line 28 performs:
```python
r"^%s(?P<path>.*)$" % re.escape(prefix.lstrip("/"))
```

This creates an inconsistency:
1. `static('')` raises `ImproperlyConfigured` (correct)
2. `static('/')` passes validation but becomes effectively empty after `lstrip('/')`
3. The resulting pattern `^(?P<path>.*)$` matches all paths, which is likely unintended
4. This violates the "Empty static prefix not permitted" contract

## Fix

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