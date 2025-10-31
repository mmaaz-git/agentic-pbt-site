# Bug Report: django.forms.ErrorDict copy() Fails to Preserve Renderer Attribute

**Target**: `django.forms.utils.ErrorDict.copy()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

ErrorDict.copy() returns a plain dict object that lacks the renderer attribute, breaking the object's rendering functionality. This occurs because ErrorDict doesn't override dict.copy(), unlike its sibling class ErrorList which correctly preserves attributes.

## Property-Based Test

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test')
    django.setup()

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

if __name__ == "__main__":
    test_errordict_copy_preserves_renderer()
```

<details>

<summary>
**Failing input**: `error_data={}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 33, in <module>
    test_errordict_copy_preserves_renderer()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 13, in test_errordict_copy_preserves_renderer
    st.dictionaries(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 29, in test_errordict_copy_preserves_renderer
    assert hasattr(copied, 'renderer'), "Copy should have renderer attribute"
           ~~~~~~~^^^^^^^^^^^^^^^^^^^^
AssertionError: Copy should have renderer attribute
Falsifying example: test_errordict_copy_preserves_renderer(
    error_data={},
)
```
</details>

## Reproducing the Bug

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test')
    django.setup()

from django.forms.utils import ErrorDict, ErrorList
from django.forms.renderers import get_default_renderer

# Create an ErrorDict with a custom renderer
renderer = get_default_renderer()
error_dict = ErrorDict(
    {'field1': ErrorList(['Error 1', 'Error 2']), 'field2': ErrorList(['Error 3'])},
    renderer=renderer
)

# Verify the original has the renderer
print("Original ErrorDict has renderer attribute:", hasattr(error_dict, 'renderer'))
print("Original ErrorDict renderer value:", error_dict.renderer)

# Create a copy using the copy() method
copied = error_dict.copy()

# Check if the copy has the renderer attribute
print("\nCopied ErrorDict has renderer attribute:", hasattr(copied, 'renderer'))

# This will raise an AttributeError
try:
    print("Copied ErrorDict renderer value:", copied.renderer)
except AttributeError as e:
    print("AttributeError accessing copied.renderer:", e)

# Verify the data was copied correctly
print("\nOriginal ErrorDict data:", dict(error_dict))
print("Copied ErrorDict data:", dict(copied))

# Show that ErrorList correctly preserves renderer on copy
error_list = ErrorList(['Error 1', 'Error 2'], renderer=renderer)
copied_list = error_list.copy()
print("\nErrorList copy preserves renderer:", hasattr(copied_list, 'renderer'))
print("ErrorList copied renderer value:", copied_list.renderer)
```

<details>

<summary>
AttributeError when accessing copied.renderer
</summary>
```
Original ErrorDict has renderer attribute: True
Original ErrorDict renderer value: <django.forms.renderers.DjangoTemplates object at 0x7d935ef7fb60>

Copied ErrorDict has renderer attribute: False
AttributeError accessing copied.renderer: 'dict' object has no attribute 'renderer'

Original ErrorDict data: {'field1': ['Error 1', 'Error 2'], 'field2': ['Error 3']}
Copied ErrorDict data: {'field1': ['Error 1', 'Error 2'], 'field2': ['Error 3']}

ErrorList copy preserves renderer: True
ErrorList copied renderer value: <django.forms.renderers.DjangoTemplates object at 0x7d935ef7fb60>
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Inconsistent API**: ErrorList, which is the sibling class in the same module with the same purpose and structure, correctly overrides copy() to preserve its renderer attribute (lines 163-167 in django/forms/utils.py). Users reasonably expect ErrorDict to behave the same way.

2. **Essential Attribute Lost**: The renderer attribute is not optional - it's required for ErrorDict's rendering functionality. The __init__ method explicitly accepts and stores this renderer (line 126), and all rendering methods depend on it through the RenderableErrorMixin inheritance.

3. **Type Inconsistency**: The copy() method returns a plain dict instead of an ErrorDict instance. This breaks the Liskov Substitution Principle - the copied object cannot be used wherever the original was used.

4. **Silent Failure**: The bug doesn't raise an error during copy(), but causes AttributeError later when trying to use rendering methods on the copied object.

## Relevant Context

The bug exists in Django's forms framework, specifically in the ErrorDict class at `/django/forms/utils.py`. The class hierarchy shows:
- ErrorDict inherits from both `dict` and `RenderableErrorMixin`
- RenderableErrorMixin provides rendering functionality that depends on the renderer attribute
- ErrorList (same module) correctly implements copy() to preserve renderer

The renderer attribute is used by Django's template rendering system to determine how to render the errors in different formats (HTML, text, JSON). Without it, the ErrorDict cannot be properly rendered in templates.

Documentation reference: https://docs.djangoproject.com/en/stable/ref/forms/api/#django.forms.Form.errors

## Proposed Fix

```diff
--- a/django/forms/utils.py
+++ b/django/forms/utils.py
@@ -130,6 +130,11 @@ class ErrorDict(dict, RenderableErrorMixin):
     def get_json_data(self, escape_html=False):
         return {f: e.get_json_data(escape_html) for f, e in self.items()}

+    def copy(self):
+        copy = self.__class__(super().copy())
+        copy.renderer = self.renderer
+        return copy
+
     def get_context(self):
         return {
             "errors": self.items(),
```