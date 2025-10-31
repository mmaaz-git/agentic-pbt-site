# Bug Report: ModelFormMixin.get_success_url() Raises Unhelpful KeyError for Missing Format Placeholders

**Target**: `django.views.generic.edit.ModelFormMixin.get_success_url()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`ModelFormMixin.get_success_url()` crashes with an unhelpful bare `KeyError` when the `success_url` template contains format placeholders that don't exist in the model object's `__dict__`, making debugging difficult for developers.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for ModelFormMixin.get_success_url() bug"""

from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import Mock
from django.views.generic.edit import ModelFormMixin


# Strategy for generating URL templates with placeholders
url_template_with_placeholders = st.builds(
    lambda prefix, placeholder, suffix: f"{prefix}{{{placeholder}}}{suffix}",
    prefix=st.text(min_size=1, max_size=20, alphabet=st.characters(categories=('Lu', 'Ll', 'Nd'), include_characters='/-')),
    placeholder=st.text(min_size=1, max_size=15, alphabet=st.characters(categories=('Lu', 'Ll', 'Nd'), include_characters='_')),
    suffix=st.text(min_size=0, max_size=20, alphabet=st.characters(categories=('Lu', 'Ll', 'Nd'), include_characters='/-'))
)

@given(success_url_template=url_template_with_placeholders)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_modelformmixin_should_not_raise_confusing_keyerror(success_url_template):
    mixin = ModelFormMixin()
    mixin.success_url = success_url_template
    mock_obj = Mock()
    mock_obj.__dict__ = {}  # Empty dict means placeholder won't be found
    mixin.object = mock_obj

    try:
        result = mixin.get_success_url()
    except KeyError as e:
        raise AssertionError(
            f"get_success_url() should not raise KeyError for URL {success_url_template!r}. "
            "It should either validate the template or provide a helpful error message."
        )

if __name__ == "__main__":
    # Run the test
    test_modelformmixin_should_not_raise_confusing_keyerror()
```

<details>

<summary>
**Failing input**: `success_url_template='0{A}'`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/52
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_modelformmixin_should_not_raise_confusing_keyerror FAILED  [100%]

=================================== FAILURES ===================================
___________ test_modelformmixin_should_not_raise_confusing_keyerror ____________
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 18, in test_modelformmixin_should_not_raise_confusing_keyerror
  |     @settings(suppress_health_check=[HealthCheck.filter_too_much])
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 27, in test_modelformmixin_should_not_raise_confusing_keyerror
    |     result = mixin.get_success_url()
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/views/generic/edit.py", line 120, in get_success_url
    |     url = self.success_url.format(**self.object.__dict__)
    | KeyError: 'A'
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 29, in test_modelformmixin_should_not_raise_confusing_keyerror
    |     raise AssertionError(
    |     ...<2 lines>...
    |     )
    | AssertionError: get_success_url() should not raise KeyError for URL '0{A}'. It should either validate the template or provide a helpful error message.
    | Falsifying example: test_modelformmixin_should_not_raise_confusing_keyerror(
    |     success_url_template='0{A}',  # or any other generated value
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 27, in test_modelformmixin_should_not_raise_confusing_keyerror
    |     result = mixin.get_success_url()
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/views/generic/edit.py", line 120, in get_success_url
    |     url = self.success_url.format(**self.object.__dict__)
    | IndexError: Replacement index 0 out of range for positional args tuple
    | Falsifying example: test_modelformmixin_should_not_raise_confusing_keyerror(
    |     success_url_template='0{0}',
    | )
    +------------------------------------
=========================== short test summary info ============================
FAILED hypo.py::test_modelformmixin_should_not_raise_confusing_keyerror - Exc...
============================== 1 failed in 0.97s ===============================
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction case for ModelFormMixin.get_success_url() KeyError bug"""

from unittest.mock import Mock
from django.views.generic.edit import ModelFormMixin

# Create a ModelFormMixin instance
mixin = ModelFormMixin()

# Set a success_url with a placeholder
mixin.success_url = "/object/{id}/success"

# Create a mock object with an empty __dict__ (no 'id' attribute)
mock_obj = Mock()
mock_obj.__dict__ = {}
mixin.object = mock_obj

# This should raise a KeyError
try:
    result = mixin.get_success_url()
    print(f"Success URL: {result}")
except KeyError as e:
    print(f"KeyError raised: {e}")
    print(f"Error type: {type(e)}")
    print(f"Error args: {e.args}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
KeyError: 'id' raised with no context about where the error occurred
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/repo.py", line 20, in <module>
    result = mixin.get_success_url()
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/views/generic/edit.py", line 120, in get_success_url
    url = self.success_url.format(**self.object.__dict__)
KeyError: 'id'
KeyError raised: 'id'
Error type: <class 'KeyError'>
Error args: ('id',)
```
</details>

## Why This Is A Bug

This violates expected Django behavior in several critical ways:

1. **Inconsistent Error Handling**: The same `get_success_url()` method uses `ImproperlyConfigured` exception at line 125-128 when `get_absolute_url()` is missing, but uses bare `KeyError` for format placeholder issues at line 120. This inconsistency makes the framework unpredictable.

2. **Unhelpful Error Message**: The bare `KeyError: 'id'` provides no context about:
   - Which view class triggered the error
   - What the full URL template was
   - What fields are actually available on the model
   - That this is a configuration issue, not a runtime data problem

3. **Contradicts Django Documentation**: The Django documentation explicitly states that `success_url` supports "dictionary string formatting interpolated against the object's field attributes" as a feature, but doesn't warn about or handle the error case when placeholders don't match.

4. **Poor Developer Experience**: This commonly occurs in real scenarios:
   - When a model uses a custom primary key field name (e.g., `uuid` instead of `id`)
   - When developers make typos in placeholder names
   - When refactoring model fields but forgetting to update URL templates
   - When copying code between views with different model structures

5. **Against Django Philosophy**: Django prides itself on helpful error messages that guide developers to solutions. Compare this to other Django errors like `FieldDoesNotExist` or `ImproperlyConfigured` which provide context and suggestions.

## Relevant Context

The bug occurs at line 120 of `/django/views/generic/edit.py`:
```python
def get_success_url(self):
    """Return the URL to redirect to after processing a valid form."""
    if self.success_url:
        url = self.success_url.format(**self.object.__dict__)  # Line 120 - No error handling!
    else:
        try:
            url = self.object.get_absolute_url()
        except AttributeError:
            raise ImproperlyConfigured(  # Proper error handling here!
                "No URL to redirect to.  Either provide a url or define"
                " a get_absolute_url method on the Model."
            )
    return url
```

Note how the `else` branch properly handles the `AttributeError` with a helpful `ImproperlyConfigured` exception, while the format string path has no error handling.

The same issue also exists in `DeletionMixin.get_success_url()` at line 236 of the same file.

Django documentation reference: https://docs.djangoproject.com/en/stable/ref/class-based-views/mixins-editing/#django.views.generic.edit.ModelFormMixin.success_url

## Proposed Fix

```diff
--- a/django/views/generic/edit.py
+++ b/django/views/generic/edit.py
@@ -117,7 +117,16 @@ class ModelFormMixin(FormMixin, SingleObjectMixin):
     def get_success_url(self):
         """Return the URL to redirect to after processing a valid form."""
         if self.success_url:
-            url = self.success_url.format(**self.object.__dict__)
+            try:
+                url = self.success_url.format(**self.object.__dict__)
+            except (KeyError, IndexError) as e:
+                available_attrs = list(self.object.__dict__.keys())
+                model_name = self.object.__class__.__name__
+                raise ImproperlyConfigured(
+                    f"Could not format success_url '{self.success_url}'. "
+                    f"Error: {e}. "
+                    f"Available attributes on {model_name}: {available_attrs}"
+                )
         else:
             try:
                 url = self.object.get_absolute_url()
@@ -233,7 +242,16 @@ class DeletionMixin:

     def get_success_url(self):
         if self.success_url:
-            return self.success_url.format(**self.object.__dict__)
+            try:
+                return self.success_url.format(**self.object.__dict__)
+            except (KeyError, IndexError) as e:
+                available_attrs = list(self.object.__dict__.keys())
+                model_name = self.object.__class__.__name__
+                raise ImproperlyConfigured(
+                    f"Could not format success_url '{self.success_url}'. "
+                    f"Error: {e}. "
+                    f"Available attributes on {model_name}: {available_attrs}"
+                )
         else:
             raise ImproperlyConfigured("No URL to redirect to. Provide a success_url.")
```