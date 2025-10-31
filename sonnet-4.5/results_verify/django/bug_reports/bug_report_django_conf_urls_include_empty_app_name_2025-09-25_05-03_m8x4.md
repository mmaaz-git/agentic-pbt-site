# Bug Report: django.conf.urls.include() Empty String app_name Treated as None

**Target**: `django.conf.urls.include()`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When calling `include((urlconf, ''), namespace='ns')` with an empty string as `app_name`, the function incorrectly raises `ImproperlyConfigured`, treating the empty string as if no `app_name` was provided. This is due to using a truthiness check (`not app_name`) instead of explicitly checking for `None`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.conf.urls import include


@given(st.text(min_size=1))
def test_empty_string_app_name_should_be_valid_with_namespace(namespace):
    patterns = []

    result = include((patterns, ''), namespace=namespace)
    urlconf_module, app_name, ns = result

    assert app_name == ''
    assert ns == namespace
```

**Failing input**: `namespace='my_namespace'` (or any non-empty string)

## Reproducing the Bug

```python
from django.conf.urls import include
from django.core.exceptions import ImproperlyConfigured

patterns = []

try:
    result = include((patterns, ''), namespace='my_namespace')
except ImproperlyConfigured:
    print("Bug confirmed: empty string app_name is rejected")
```

## Why This Is A Bug

Python's truthiness semantics treat empty strings as falsy (`not '' == True`), but an empty string is a distinct value from `None`. When a user explicitly provides `''` as an `app_name`, it should be honored as a valid (if unusual) value, not conflated with the absence of an `app_name`. The validation logic should use `app_name is None` to check for the absence of a value, not `not app_name` which also rejects empty strings.

## Fix

```diff
--- a/django/urls/conf.py
+++ b/django/urls/conf.py
@@ -39,7 +39,7 @@ def include(arg, namespace=None):
         urlconf_module = import_module(urlconf_module)
     patterns = getattr(urlconf_module, "urlpatterns", urlconf_module)
     app_name = getattr(urlconf_module, "app_name", app_name)
-    if namespace and not app_name:
+    if namespace and app_name is None:
         raise ImproperlyConfigured(
             "Specifying a namespace in include() without providing an app_name "
             "is not supported. Set the app_name attribute in the included "
```