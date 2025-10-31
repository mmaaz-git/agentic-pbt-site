# Bug Report: ModelFormMixin.get_success_url() KeyError on Mismatched Format Placeholders

**Target**: `django.views.generic.edit.ModelFormMixin.get_success_url()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`ModelFormMixin.get_success_url()` crashes with an unhelpful `KeyError` when the `success_url` template contains format placeholders (e.g., `{id}`) that don't exist in `self.object.__dict__`. This occurs at line 120 in `django/views/generic/edit.py` where `self.success_url.format(**self.object.__dict__)` is executed without validation or error handling.

## Property-Based Test

```python
from hypothesis import given, assume, strategies as st
from unittest.mock import Mock
from django.views.generic.edit import ModelFormMixin


@given(success_url_template=st.text(min_size=1, max_size=100))
def test_modelformmixin_should_not_raise_confusing_keyerror(success_url_template):
    assume('{' in success_url_template and '}' in success_url_template)

    mixin = ModelFormMixin()
    mixin.success_url = success_url_template
    mock_obj = Mock()
    mock_obj.__dict__ = {}
    mixin.object = mock_obj

    try:
        result = mixin.get_success_url()
    except KeyError as e:
        raise AssertionError(
            f"get_success_url() should not raise KeyError for URL {success_url_template!r}. "
            "It should either validate the template or provide a helpful error message."
        )
```

**Failing input**: `success_url = "/object/{id}/success"` with empty `object.__dict__`

## Reproducing the Bug

```python
from unittest.mock import Mock
from django.views.generic.edit import ModelFormMixin

mixin = ModelFormMixin()
mixin.success_url = "/object/{id}/success"
mock_obj = Mock()
mock_obj.__dict__ = {}
mixin.object = mock_obj

result = mixin.get_success_url()
```

Running this code raises:
```
KeyError: 'id'
```

## Why This Is A Bug

1. **Unhelpful error message**: The raw `KeyError` doesn't explain that the URL template requires certain attributes on the model object
2. **No validation**: The code doesn't validate that the URL template's placeholders match available object attributes
3. **Inconsistent with Django patterns**: Other Django code raises `ImproperlyConfigured` for misconfiguration issues
4. **Poor developer experience**: When a developer uses a placeholder that doesn't exist on their model, they get a confusing error
5. **Affects real usage**: This commonly occurs when:
   - A model doesn't have a field referenced in success_url
   - A model uses a custom primary key field name
   - A developer makes a typo in the template

The code at line 120 performs `self.success_url.format(**self.object.__dict__)` without any try/except or validation.

## Fix

Add proper error handling to provide a helpful error message:

```diff
--- a/django/views/generic/edit.py
+++ b/django/views/generic/edit.py
@@ -117,7 +117,14 @@ class ModelFormMixin(FormMixin, SingleObjectMixin):
     def get_success_url(self):
         """Return the URL to redirect to after processing a valid form."""
         if self.success_url:
-            url = self.success_url.format(**self.object.__dict__)
+            try:
+                url = self.success_url.format(**self.object.__dict__)
+            except KeyError as e:
+                raise ImproperlyConfigured(
+                    f"success_url {self.success_url!r} contains placeholder {e} "
+                    f"that doesn't exist in {self.object.__class__.__name__}.__dict__. "
+                    f"Available fields: {list(self.object.__dict__.keys())}"
+                )
         else:
             try:
                 url = self.object.get_absolute_url()
```