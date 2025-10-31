# Bug Report: django.forms.ErrorDict copy() doesn't preserve renderer

**Target**: `django.forms.utils.ErrorDict.copy()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

ErrorDict.copy() fails to preserve the `renderer` attribute because it inherits dict.copy() which only performs a shallow copy of dictionary data, not custom attributes.

## Property-Based Test

```python
from django.forms.utils import ErrorDict, ErrorList
from django.forms.renderers import get_default_renderer
from hypothesis import given, strategies as st

@given(
    st.dictionaries(
        st.text(min_size=1, max_size=50),
        st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=5).map(
            lambda errors: ErrorList(errors)
        ),
        min_size=0,
        max_size=10
    )
)
def test_errordict_copy_preserves_renderer(error_data):
    renderer = get_default_renderer()

    error_dict = ErrorDict(error_data, renderer=renderer)

    copied = error_dict.copy()

    assert hasattr(copied, 'renderer'), "Copy should have renderer attribute"
    assert copied.renderer == error_dict.renderer, "Copy should preserve renderer"
```

**Failing input**: `error_data={}` (or any dictionary)

## Reproducing the Bug

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test')
    django.setup()

from django.forms.utils import ErrorDict, ErrorList
from django.forms.renderers import get_default_renderer

renderer = get_default_renderer()

error_dict = ErrorDict(
    {'field1': ErrorList(['Error 1'])},
    renderer=renderer
)

copied = error_dict.copy()

assert hasattr(error_dict, 'renderer')
assert not hasattr(copied, 'renderer')
```

## Why This Is A Bug

ErrorDict.__init__ accepts a `renderer` parameter and stores it as an instance attribute. However, ErrorDict extends dict and doesn't override copy(), so it uses dict.copy() which only creates a shallow copy of the dictionary keys/values, not custom attributes. This violates user expectations - copying an ErrorDict should preserve all its attributes, not just the dictionary data.

## Fix

```diff
--- a/django/forms/utils.py
+++ b/django/forms/utils.py
@@ -125,6 +125,11 @@ class ErrorDict(dict, RenderableErrorMixin):
     def as_data(self):
         return {f: e.as_data() for f, e in self.items()}

+    def copy(self):
+        copy = super().copy()
+        copy.renderer = self.renderer
+        return copy
+
     def get_json_data(self, escape_html=False):
         return {f: e.get_json_data(escape_html) for f, e in self.items()}
```