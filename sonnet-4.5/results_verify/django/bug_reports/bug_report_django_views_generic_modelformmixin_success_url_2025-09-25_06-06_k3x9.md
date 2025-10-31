# Bug Report: Django ModelFormMixin and DeletionMixin KeyError on Missing success_url Format Parameters

**Target**: `django.views.generic.edit.ModelFormMixin.get_success_url` and `django.views.generic.edit.DeletionMixin.get_success_url`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`ModelFormMixin.get_success_url()` and `DeletionMixin.get_success_url()` raise unhandled `KeyError` when the `success_url` format string contains placeholders that don't exist in `self.object.__dict__`, causing 500 errors instead of providing clear configuration error messages.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from django.views.generic.edit import ModelFormMixin, DeletionMixin

@st.composite
def url_template_with_placeholders(draw):
    num_placeholders = draw(st.integers(min_value=1, max_value=5))
    placeholders = [draw(st.text(min_size=1, max_size=20).filter(lambda x: x.isidentifier()))
                    for _ in range(num_placeholders)]
    template_parts = ['/redirect']
    for placeholder in placeholders:
        template_parts.append(f'/{{{placeholder}}}')
    template_parts.append('/')
    return ''.join(template_parts)

@given(
    url_template_with_placeholders(),
    st.dictionaries(
        st.text(min_size=1, max_size=20).filter(lambda x: x.isidentifier()),
        st.one_of(st.text(min_size=0, max_size=100), st.integers(), st.none()),
        min_size=1, max_size=10
    )
)
@settings(max_examples=500)
def test_success_url_format_with_object_dict(url_template, object_dict):
    class MockObject:
        def __init__(self, attrs):
            self.__dict__.update(attrs)

    class TestModelFormMixin(ModelFormMixin):
        success_url = url_template

    view = TestModelFormMixin()
    view.object = MockObject(object_dict)

    url = view.get_success_url()
```

**Failing input**: `url_template='/redirect/{B}/'`, `object_dict={'LW': None}`

## Reproducing the Bug

```python
import django
from django.conf import settings

settings.configure(
    DEBUG=True,
    SECRET_KEY='test-key',
    DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
    INSTALLED_APPS=['django.contrib.contenttypes', 'django.contrib.auth'],
)
django.setup()

from django.views.generic import UpdateView, DeleteView
from django.db import models

class Article(models.Model):
    title = models.CharField(max_length=100)
    slug = models.SlugField()
    class Meta:
        app_label = 'test'

class ArticleUpdateView(UpdateView):
    model = Article
    fields = ['title']
    success_url = '/articles/{category}/{id}/'

article = Article(id=1, title='Test', slug='test')
article.__dict__.update({'id': 1})

update_view = ArticleUpdateView()
update_view.object = article

try:
    url = update_view.get_success_url()
except KeyError as e:
    print(f"Bug reproduced: KeyError({e})")
```

## Why This Is A Bug

Both `ModelFormMixin.get_success_url()` (line 120 of `edit.py`) and `DeletionMixin.get_success_url()` (line 236 of `edit.py`) perform `self.success_url.format(**self.object.__dict__)` without handling potential `KeyError` exceptions when required format parameters are missing from the object's `__dict__`.

This is similar to the existing issue in `RedirectView.get_redirect_url()` but affects different view classes. When a developer configures a `success_url` with format placeholders that don't match the model's attributes, the view crashes with an unhelpful `KeyError` instead of raising a clear `ImproperlyConfigured` exception.

This is particularly problematic because:
1. It causes 500 errors in production instead of being caught during development
2. The error message doesn't indicate that the issue is with the `success_url` configuration
3. It's a common mistake when models are refactored and success URLs aren't updated

## Fix

```diff
diff --git a/django/views/generic/edit.py b/django/views/generic/edit.py
index 1234567..abcdefg 100644
--- a/django/views/generic/edit.py
+++ b/django/views/generic/edit.py
@@ -117,7 +117,13 @@ class ModelFormMixin(FormMixin, SingleObjectMixin):
     def get_success_url(self):
         """Return the URL to redirect to after processing a valid form."""
         if self.success_url:
-            url = self.success_url.format(**self.object.__dict__)
+            try:
+                url = self.success_url.format(**self.object.__dict__)
+            except (KeyError, ValueError) as e:
+                raise ImproperlyConfigured(
+                    f"success_url '{self.success_url}' could not be formatted "
+                    f"with object attributes {list(self.object.__dict__.keys())}: {e}"
+                )
         else:
             try:
                 url = self.object.get_absolute_url()
@@ -233,7 +239,13 @@ class DeletionMixin:

     def get_success_url(self):
         if self.success_url:
-            return self.success_url.format(**self.object.__dict__)
+            try:
+                return self.success_url.format(**self.object.__dict__)
+            except (KeyError, ValueError) as e:
+                raise ImproperlyConfigured(
+                    f"success_url '{self.success_url}' could not be formatted "
+                    f"with object attributes {list(self.object.__dict__.keys())}: {e}"
+                )
         else:
             raise ImproperlyConfigured("No URL to redirect to. Provide a success_url.")
```