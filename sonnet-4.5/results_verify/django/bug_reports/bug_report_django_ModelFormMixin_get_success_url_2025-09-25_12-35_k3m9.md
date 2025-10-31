# Bug Report: django.views.generic.ModelFormMixin URL Format KeyError

**Target**: `django.views.generic.edit.ModelFormMixin.get_success_url`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`ModelFormMixin.get_success_url` raises an unhandled `KeyError` when the `success_url` template contains placeholders that are not present in `self.object.__dict__`.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st, assume
from django.views.generic.edit import ModelFormMixin
from django.contrib.auth.models import User

@given(
    placeholder_key=st.text(alphabet='abcdefghijklmnopqrstuvwxyz_', min_size=1, max_size=10),
    object_attrs=st.dictionaries(
        st.text(alphabet='abcdefghijklmnopqrstuvwxyz_', min_size=1, max_size=10),
        st.text(max_size=20),
        max_size=5
    )
)
@settings(max_examples=1000)
def test_modelformmixin_success_url_formatting(placeholder_key, object_attrs):
    assume(placeholder_key not in object_attrs)

    class TestMixin(ModelFormMixin):
        def __init__(self):
            super().__init__()
            self.success_url = f'/path/{{{placeholder_key}}}/'
            self.model = User

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

from django.views.generic.edit import ModelFormMixin
from django.contrib.auth.models import User

class TestMixin(ModelFormMixin):
    def __init__(self):
        super().__init__()
        self.success_url = '/user/{user_id}/detail/'
        self.model = User

mixin = TestMixin()

class MockObject:
    pass

mixin.object = MockObject()

result = mixin.get_success_url()
```

Running this raises:
```
KeyError: 'user_id'
```

## Why This Is A Bug

The code at line 120 performs string formatting without handling the case where placeholders don't match the object's attributes. This can occur when:

1. A view is configured with a `success_url` template expecting certain object attributes
2. The object being saved doesn't have all those attributes
3. The developer makes a typo in the success_url placeholder name

The current implementation crashes with an unhelpful `KeyError` instead of providing a meaningful error message.

## Fix

```diff
--- a/django/views/generic/edit.py
+++ b/django/views/generic/edit.py
@@ -117,7 +117,11 @@ class ModelFormMixin(FormMixin, SingleObjectMixin):
     def get_success_url(self):
         """Return the URL to redirect to after processing a valid form."""
         if self.success_url:
-            url = self.success_url.format(**self.object.__dict__)
+            try:
+                url = self.success_url.format(**self.object.__dict__)
+            except KeyError as e:
+                raise ImproperlyConfigured(
+                    f"success_url template contains placeholder {e} not present in object attributes"
+                )
         else:
             try:
                 url = self.object.get_absolute_url()
```