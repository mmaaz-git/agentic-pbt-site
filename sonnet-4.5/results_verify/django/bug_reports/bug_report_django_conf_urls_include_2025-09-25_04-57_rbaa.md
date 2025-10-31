# Bug Report: django.conf.urls.include app_name Type Validation Missing

**Target**: `django.conf.urls.include`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `include()` function accepts non-string values for `app_name` in 2-tuples without validation, causing `TypeError` crashes during URL resolution when the resolver tries to join app names with ":".

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.conf.urls import include
from django.urls import path, get_resolver
from django.http import HttpResponse
from django.conf import settings
import pytest


@given(st.lists(st.text()))
def test_include_2tuple_app_name_must_be_string_or_none(non_string_value):
    def view(request):
        return HttpResponse("test")

    patterns = [path('test/', view)]

    result = include((patterns, non_string_value))

    settings.configure(
        DEBUG=True,
        ROOT_URLCONF='__main__',
        SECRET_KEY='test'
    )

    import django
    django.setup()

    urlpatterns = [path('prefix/', result)]
    resolver = get_resolver()

    with pytest.raises(TypeError):
        resolver.resolve('/prefix/test/')
```

**Failing input**: `app_name = ['not', 'a', 'string']` (any non-string, non-None value)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.conf.urls import include
from django.urls import path, get_resolver
from django.http import HttpResponse
from django.conf import settings

def test_view(request):
    return HttpResponse("test")

patterns = [path('test/', test_view)]
invalid_app_name = ['not', 'a', 'string']

result = include((patterns, invalid_app_name))

settings.configure(DEBUG=True, ROOT_URLCONF='__main__', SECRET_KEY='test')
urlpatterns = [path('prefix/', result)]

import django
django.setup()

resolver = get_resolver()
resolved = resolver.resolve('/prefix/test/')
```

Output:
```
TypeError: sequence item 0: expected str instance, list found
```

## Why This Is A Bug

The `include()` function should validate that `app_name` is either a string or `None`. When a non-string value is provided in a 2-tuple `(patterns, app_name)`, the function accepts it without validation. Later, when Django's URL resolver attempts to create a `ResolverMatch` object, it tries to join app names using `":".join(self.app_names)` (in `django/urls/resolvers.py:60`), which crashes if any app_name is not a string.

This violates the API contract - `app_name` should represent an application namespace string identifier, not arbitrary Python objects.

## Fix

```diff
--- a/django/urls/conf.py
+++ b/django/urls/conf.py
@@ -18,6 +18,9 @@ def include(arg, namespace=None):
     app_name = None
     if isinstance(arg, tuple):
         # Callable returning a namespace hint.
+        if len(arg) == 2 and arg[1] is not None and not isinstance(arg[1], str):
+            raise ImproperlyConfigured(
+                "app_name in 2-tuple must be a string or None, got %s" % type(arg[1]).__name__)
         try:
             urlconf_module, app_name = arg
         except ValueError:
```