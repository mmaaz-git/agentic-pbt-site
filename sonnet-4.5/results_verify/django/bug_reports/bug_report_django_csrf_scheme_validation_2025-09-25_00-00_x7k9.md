# Bug Report: django.core.checks.compatibility CSRF Scheme Validation Mismatch

**Target**: `django.core.checks.compatibility.django_4_0.check_csrf_trusted_origins`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `check_csrf_trusted_origins` function accepts any origin containing "://" anywhere in the string, even when the scheme is invalid or positioned incorrectly. The error message claims origins "must start with a scheme" but the code only checks if "://" exists anywhere.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from unittest.mock import patch
from urllib.parse import urlsplit
from django.core.checks.compatibility.django_4_0 import check_csrf_trusted_origins

@given(
    st.text(min_size=1, alphabet=st.characters(min_codepoint=97, max_codepoint=122)).filter(
        lambda s: "://" not in s
    ),
    st.text(min_size=1, alphabet=st.characters(min_codepoint=97, max_codepoint=122)).filter(
        lambda s: "://" not in s
    )
)
def test_scheme_must_be_at_start_not_anywhere(prefix, suffix):
    origin = f"{prefix}://{suffix}"
    parsed = urlsplit(origin)
    is_valid_scheme = parsed.scheme in ['http', 'https', 'ftp', 'ws', 'wss']

    with patch.object(settings, 'CSRF_TRUSTED_ORIGINS', [origin]):
        result = check_csrf_trusted_origins(app_configs=None)
        if not is_valid_scheme:
            assert len(result) > 0, f"Origin '{origin}' should be rejected but wasn't"
```

**Failing input**: `prefix='a'`, `suffix='a'` (produces origin `'a://a'`)

## Reproducing the Bug

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test', CSRF_TRUSTED_ORIGINS=[])
    django.setup()

from django.core.checks.compatibility.django_4_0 import check_csrf_trusted_origins
from unittest.mock import patch
from urllib.parse import urlsplit

origin = "a://a"

with patch.object(settings, 'CSRF_TRUSTED_ORIGINS', [origin]):
    errors = check_csrf_trusted_origins(app_configs=None)
    print(f"Errors for '{origin}': {len(errors)}")

parsed = urlsplit(origin)
print(f"urlsplit interprets scheme as: '{parsed.scheme}', netloc as: '{parsed.netloc}'")
```

Output:
```
Errors for 'a://a': 0
urlsplit interprets scheme as: 'a', netloc as: 'a'
```

## Why This Is A Bug

The error message explicitly states that CSRF_TRUSTED_ORIGINS values "must start with a scheme (usually http:// or https://)", but the validation logic only checks if "://" exists anywhere in the string:

```python
if "://" not in origin:
    errors.append(Error(...))
```

This allows malformed origins like:
- `"a://a"` - invalid scheme
- `"example.com://http"` - scheme in wrong position
- `"://example.com"` - empty scheme

When the middleware uses `urlsplit()` on these malformed origins, they are misparsed, potentially affecting CSRF protection logic.

## Fix

```diff
--- a/django/core/checks/compatibility/django_4_0.py
+++ b/django/core/checks/compatibility/django_4_0.py
@@ -1,4 +1,5 @@
 from django.conf import settings
+from urllib.parse import urlsplit

 from .. import Error, Tags, register

@@ -7,7 +8,10 @@ from .. import Error, Tags, register
 def check_csrf_trusted_origins(app_configs, **kwargs):
     errors = []
     for origin in settings.CSRF_TRUSTED_ORIGINS:
-        if "://" not in origin:
+        parsed = urlsplit(origin)
+        if not parsed.scheme or not parsed.netloc:
             errors.append(
                 Error(
                     "As of Django 4.0, the values in the CSRF_TRUSTED_ORIGINS "
```

Alternatively, validate against a list of known valid schemes:
```python
VALID_SCHEMES = {'http', 'https'}
parsed = urlsplit(origin)
if parsed.scheme not in VALID_SCHEMES or not parsed.netloc:
    errors.append(...)
```