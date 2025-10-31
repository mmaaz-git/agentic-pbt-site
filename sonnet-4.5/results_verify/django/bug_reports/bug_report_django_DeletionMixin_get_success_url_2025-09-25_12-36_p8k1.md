# Bug Report: django.views.generic.DeletionMixin URL Format KeyError

**Target**: `django.views.generic.edit.DeletionMixin.get_success_url`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`DeletionMixin.get_success_url` raises an unhandled `KeyError` when the `success_url` template contains placeholders that are not present in `self.object.__dict__`.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st, assume
from django.views.generic.edit import DeletionMixin

@given(
    placeholder_key=st.text(alphabet='abcdefghijklmnopqrstuvwxyz_', min_size=1, max_size=10),
    object_attrs=st.dictionaries(
        st.text(alphabet='abcdefghijklmnopqrstuvwxyz_', min_size=1, max_size=10),
        st.text(max_size=20),
        max_size=5
    )
)
@settings(max_examples=1000)
def test_deletionmixin_success_url_formatting(placeholder_key, object_attrs):
    assume(placeholder_key not in object_attrs)

    class TestMixin(DeletionMixin):
        def __init__(self):
            super().__init__()
            self.success_url = f'/path/{{{placeholder_key}}}/'

    mixin = TestMixin()

    class MockObject:
        def __init__(self, attrs):
            self.__dict__.update(attrs)

    mixin.object = MockObject(object_attrs)
    result = mixin.get_success_url()
```

**Failing input**: `placeholder_key='_'`, `object_attrs={}`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

import django
from django.conf import settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
        INSTALLED_APPS=['django.contrib.contenttypes', 'django.contrib.auth'],
        USE_TZ=True,
    )
    django.setup()

from django.views.generic.edit import DeletionMixin

class TestMixin(DeletionMixin):
    def __init__(self):
        super().__init__()
        self.success_url = '/category/{category_id}/'

mixin = TestMixin()

class MockObject:
    id = 42

mixin.object = MockObject()

result = mixin.get_success_url()
```

Running this raises:
```
KeyError: 'category_id'
```

## Why This Is A Bug

The code at line 236 performs string formatting without handling the case where placeholders don't match the object's attributes. This can occur when:

1. A DeleteView is configured with a `success_url` template expecting certain object attributes
2. The object being deleted doesn't have all those attributes
3. The developer makes a typo in the success_url placeholder name

The current implementation crashes with an unhelpful `KeyError` instead of providing a meaningful error message.

## Fix

```diff
--- a/django/views/generic/edit.py
+++ b/django/views/generic/edit.py
@@ -233,7 +233,11 @@ class DeletionMixin:

     def get_success_url(self):
         if self.success_url:
-            return self.success_url.format(**self.object.__dict__)
+            try:
+                return self.success_url.format(**self.object.__dict__)
+            except KeyError as e:
+                raise ImproperlyConfigured(
+                    f"success_url template contains placeholder {e} not present in object attributes"
+                )
         else:
             raise ImproperlyConfigured("No URL to redirect to. Provide a success_url.")
```