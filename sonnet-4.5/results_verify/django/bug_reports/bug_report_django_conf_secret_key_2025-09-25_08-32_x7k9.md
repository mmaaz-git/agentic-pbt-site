# Bug Report: django.conf SECRET_KEY Misleading Error Message

**Target**: `django.conf.LazySettings.__getattr__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When `SECRET_KEY` is set to a non-empty falsy value (e.g., `False`, `0`), the error message incorrectly states "must not be empty" even though the value is not empty in the traditional sense.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.conf import LazySettings
from django.core.exceptions import ImproperlyConfigured
import pytest


@given(st.one_of(st.just(False), st.just(0), st.just([]), st.just({})))
def test_secret_key_non_empty_falsy_misleading_error(falsy_value):
    settings_obj = LazySettings()
    settings_obj.configure(SECRET_KEY=falsy_value)

    with pytest.raises(ImproperlyConfigured) as exc_info:
        _ = settings_obj.SECRET_KEY

    error_msg = str(exc_info.value)
    assert "must not be empty" in error_msg
```

**Failing input**: `falsy_value=False` (or `0`, `[]`, `{}`)

## Reproducing the Bug

```python
from django.conf import LazySettings
from django.core.exceptions import ImproperlyConfigured

settings = LazySettings()
settings.configure(SECRET_KEY=False)

try:
    _ = settings.SECRET_KEY
except ImproperlyConfigured as e:
    print(f"Error: {e}")
```

Output:
```
Error: The SECRET_KEY setting must not be empty.
```

## Why This Is A Bug

The error message is misleading because:
- `False` is a valid boolean value, not "empty"
- `0` is a valid integer, not "empty"
- The error message should accurately describe what values are not allowed

The code at line 89-90 uses a truthiness check (`not val`) but the error message only mentions "empty":

```python
elif name == "SECRET_KEY" and not val:
    raise ImproperlyConfigured("The SECRET_KEY setting must not be empty.")
```

This check rejects all falsy values (empty string, None, False, 0, [], {}, etc.), but the error message only makes sense for empty strings and possibly None.

## Fix

```diff
--- a/django/conf/__init__.py
+++ b/django/conf/__init__.py
@@ -87,7 +87,7 @@ class LazySettings(LazyObject):
         if name in {"MEDIA_URL", "STATIC_URL"} and val is not None:
             val = self._add_script_prefix(val)
         elif name == "SECRET_KEY" and not val:
-            raise ImproperlyConfigured("The SECRET_KEY setting must not be empty.")
+            raise ImproperlyConfigured("The SECRET_KEY setting must not be empty or falsy (got %s)." % type(val).__name__)

         self.__dict__[name] = val
         return val
```

Alternatively, if the intention is to only disallow empty strings and None, the check should be more specific:

```diff
--- a/django/conf/__init__.py
+++ b/django/conf/__init__.py
@@ -87,7 +87,7 @@ class LazySettings(LazyObject):
         if name in {"MEDIA_URL", "STATIC_URL"} and val is not None:
             val = self._add_script_prefix(val)
-        elif name == "SECRET_KEY" and not val:
+        elif name == "SECRET_KEY" and (val is None or val == ""):
             raise ImproperlyConfigured("The SECRET_KEY setting must not be empty.")

         self.__dict__[name] = val
```