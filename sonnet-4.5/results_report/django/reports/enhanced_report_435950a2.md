# Bug Report: Django ModelFormMixin and DeletionMixin KeyError on Missing success_url Format Parameters

**Target**: `django.views.generic.edit.ModelFormMixin.get_success_url` and `django.views.generic.edit.DeletionMixin.get_success_url`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`ModelFormMixin.get_success_url()` and `DeletionMixin.get_success_url()` crash with unhandled `KeyError` when the `success_url` format string contains placeholders that don't exist in `self.object.__dict__`, causing 500 errors in production instead of raising clear configuration error messages during development.

## Property-Based Test

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

from hypothesis import given, strategies as st, settings as hypo_settings
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
@hypo_settings(max_examples=500)
def test_success_url_format_with_object_dict(url_template, object_dict):
    class MockObject:
        def __init__(self, attrs):
            self.__dict__.update(attrs)

    class TestModelFormMixin(ModelFormMixin):
        success_url = url_template

    view = TestModelFormMixin()
    view.object = MockObject(object_dict)

    try:
        url = view.get_success_url()
        print(f"✓ Success URL generated: {url}")
    except KeyError as e:
        print(f"✗ KeyError raised for template '{url_template}' with object_dict {object_dict}: {e}")
        raise

if __name__ == "__main__":
    print("Running Hypothesis test for ModelFormMixin.get_success_url()")
    print("=" * 60)
    try:
        test_success_url_format_with_object_dict()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed!")
        import traceback
        traceback.print_exc()
```

<details>

<summary>
**Failing input**: `url_template='/redirect/{B}/', object_dict={'m': None}`
</summary>
```
Running Hypothesis test for ModelFormMixin.get_success_url()
============================================================
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'Å': 57416697497134083382136962890481153953}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'Å': 57416697497134083382136962890481153953}: 'else'
✓ Success URL generated: /redirect/None/
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'Å': 57416697497134083382136962890481153953}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'Å': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'Å': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'Å': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'Å': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'Å': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'Å': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'Å': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'Å': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'Å': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'Å': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'Å': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'Å': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'Å': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'Å': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'b': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'a': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'Q': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'O': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'M': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'K': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'I': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'G': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'C': ''}: 'else'
✗ KeyError raised for template '/redirect/{else}/' with object_dict {'A': ''}: 'else'
✗ KeyError raised for template '/redirect/{els}/' with object_dict {'A': ''}: 'els'
✗ KeyError raised for template '/redirect/{el}/' with object_dict {'A': ''}: 'el'
✗ KeyError raised for template '/redirect/{e}/' with object_dict {'A': ''}: 'e'
✗ KeyError raised for template '/redirect/{U}/' with object_dict {'A': ''}: 'U'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'A': ''}: 'B'
✓ Success URL generated: /redirect//
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'à': '\x94\U0010c249\x84\tø\U000c59e4\U00071884', 'cHU': -28}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'òñ': None, 'ºQ': 114, 'Z': 32320834663674298263462438531036400041, 'á': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'t': None, 'è': -6899417662782219710161919647790397733, 'i': -22704, 'iç彑': None, 'AD': None, 'b': 1039, 'NAME': None, 'uR': -1361746092, 'Y': 364265463, 'D': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'ê': '', 'ð': None, 'Inf': '\U000beb28Ê'}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'ë': 'n\x97Ã'}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'ª': 'Ã', '__main__': 500}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'else': -6845, 'ð': 29138}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'Æi': ''}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'k': 28274}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'böØ': 'A', 'p': ''}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'Ñ': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'ï': '\x974'}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'S': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'default': 0, 'ÛÁ': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'Òi': None, 'NULL': ''}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'E': None, 'á': -86, 'NIL': 47}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'ENGINE': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'y': None, 'null': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'L': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'nil': 89, 'Fî': '__main__', 'Å': '\x9f', 'ò': ',¹\U0001b737¼𞁊', 'none': 113, 'NUL': 'È<©\U00103a15𨇧r'}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'ëGö': 12847}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'Ô': 'Ò\x1fóÖ'}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'FALSE': 0}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'c': None, 'P': 12005, '𫹛': None, 'ø': None, '𡸷': '\x98ô»', 'ê': None, 'Q': 'Ç¿5AÌ\U000ed8a2', 'Ï': None, 'íß': '\U000a59b4¸¨ö\x9aÙ-', 'Ä': "°'\U000992eeu"}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'þðVö': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'åæ': 'Ý\x93MÕ\U000bcb17=+'}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'h': '\x98!O', 'U': None, 'ñx': None, 'iÅíª': '\x99\U000f20d8\U0007a234Ú', 'ãm': 500, 'undefined': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'default': 21782}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'Ⱦ': 0}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'LÇc': 13022}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'Scunthorpe': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'퍫': 27874, 'Ì': 23235, 'INF': -65, 'Æ': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'nil': None, 'I': ''}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'L': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'Õå': 11160}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'õ': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'m': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'Y': -310392610, 'q': '\U0007dca0\U000653e4\x9a\x90\x0f>ï\x86©\U000f278a\U0008f244öÓ\x80¿¼\U0010906a\x8e\x02\U000a3b46q65\x0bÈ\x8f\U000fa7bd', 'a': '\x95\U000de876'}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'__main__': 14542, 'Ü': -86, 'Scunthorpe': 511879819643351726}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'Eú': None, 'default': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'Ê': -125, 'if': '¬Ø\x8d'}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'k': '\U000a9381¼\U0004b4e2b¨', '__main__': -140598789494341136393037825282540055039, '𛅸Ýþ': 114, 'None': None, 'â': '𛁮¨', 'ENGINE': None, 'Ü': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'ë': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'T': '\U001026f9å\x8f+'}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'û': ''}: 'B'Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 57, in <module>
    test_success_url_format_with_object_dict()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 27, in test_success_url_format_with_object_dict
    url_template_with_placeholders(),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 47, in test_success_url_format_with_object_dict
    url = view.get_success_url()
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/views/generic/edit.py", line 120, in get_success_url
    url = self.success_url.format(**self.object.__dict__)
KeyError: 'B'
Falsifying example: test_success_url_format_with_object_dict(
    url_template='/redirect/{B}/',
    object_dict={'m': None},
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/53/hypo.py:49

✗ KeyError raised for template '/redirect/{B}/' with object_dict {'ø': 0}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'Hz': '9Ö^\U000f095d\x97\x86\x92'}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'îÛ': 'django.contrib.auth'}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'â': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'ENGINE': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'true': 0}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'ïÁ': '\x8fÆé', 'kb': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'I': 'Scunthorpe'}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'Ä': 'Ý:°{'}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'iÊ': -58260913007705684931271754647240229726, 'í': 17435, 'W': None, '鼨Ã': ':memory:'}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'L': '\x7f\x14;úRÅøJE\x07\x91è'}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'I': ''}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'WY': 'î¸\x915\U000c9601À\x97'}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'Dn': 'w', 'Û·': 22464483416773775558988982727203738202}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'Âb': '\x9b\U00105de1ì'}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'è': None}: 'B'
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'z': 'RP\x0f\x01°ÈËü¸\x0b\U000949fe\U000fef26\x7f÷m9\x97\U0009714b'}: 'B'
✓ Success URL generated: /redirect/None/
✗ KeyError raised for template '/redirect/{B}/' with object_dict {'m': None}: 'B'

Test failed!
```
</details>

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

# Test ModelFormMixin via UpdateView
class ArticleUpdateView(UpdateView):
    model = Article
    fields = ['title']
    success_url = '/articles/{category}/{id}/'  # 'category' doesn't exist in Article

article = Article(id=1, title='Test', slug='test')
article.__dict__.update({'id': 1, 'title': 'Test', 'slug': 'test'})

update_view = ArticleUpdateView()
update_view.object = article

print("Testing ModelFormMixin.get_success_url() with missing format parameter:")
try:
    url = update_view.get_success_url()
    print(f"Success URL: {url}")
except KeyError as e:
    print(f"KeyError raised: {e}")
    print(f"Exception type: {type(e).__name__}")

# Test DeletionMixin via DeleteView
class ArticleDeleteView(DeleteView):
    model = Article
    success_url = '/articles/{section}/{pk}/'  # 'section' and 'pk' don't exist

delete_view = ArticleDeleteView()
delete_view.object = article

print("\nTesting DeletionMixin.get_success_url() with missing format parameters:")
try:
    url = delete_view.get_success_url()
    print(f"Success URL: {url}")
except KeyError as e:
    print(f"KeyError raised: {e}")
    print(f"Exception type: {type(e).__name__}")
```

<details>

<summary>
KeyError exceptions are raised when success_url placeholders don't match object attributes
</summary>
```
Testing ModelFormMixin.get_success_url() with missing format parameter:
KeyError raised: 'category'
Exception type: KeyError

Testing DeletionMixin.get_success_url() with missing format parameters:
KeyError raised: 'section'
Exception type: KeyError
```
</details>

## Why This Is A Bug

This violates Django's expected behavior in several ways:

1. **Inconsistent error handling**: Django typically raises `ImproperlyConfigured` exceptions for configuration errors. Both `ModelFormMixin.get_success_url()` at line 120 and `DeletionMixin.get_success_url()` at line 236 of `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/views/generic/edit.py` perform `self.success_url.format(**self.object.__dict__)` without catching potential `KeyError` exceptions.

2. **Poor developer experience**: When a developer mistakenly uses a placeholder that doesn't exist in the model (e.g., using `{pk}` instead of `{id}`, or `{category}` for a non-existent field), they get an unhelpful `KeyError: 'pk'` instead of a clear message indicating the configuration problem.

3. **Production crashes**: This causes 500 errors in production rather than being caught during development with a clear error message about the misconfigured `success_url`.

4. **Inconsistency with other Django components**: The `RedirectView.get_redirect_url()` method in Django has similar logic but at least documents this behavior. The generic editing views don't document or handle this potential failure mode.

## Relevant Context

The Django documentation for [UpdateView](https://docs.djangoproject.com/en/stable/ref/class-based-views/generic-editing/#django.views.generic.edit.UpdateView) and [DeleteView](https://docs.djangoproject.com/en/stable/ref/class-based-views/generic-editing/#django.views.generic.edit.DeleteView) doesn't explicitly mention that `success_url` placeholders must match object attributes exactly, nor that a `KeyError` will be raised if they don't match.

This is particularly problematic when:
- Refactoring models (renaming fields)
- Copy-pasting view configurations between different models
- Using common field names that might not exist on all models (like `pk` vs `id`)
- Typos in placeholder names

The issue affects both `ModelFormMixin` (used by `CreateView` and `UpdateView`) and `DeletionMixin` (used by `DeleteView`), making it a widespread problem in Django's generic class-based views.

## Proposed Fix

```diff
diff --git a/django/views/generic/edit.py b/django/views/generic/edit.py
index abc123..def456 100644
--- a/django/views/generic/edit.py
+++ b/django/views/generic/edit.py
@@ -117,7 +117,13 @@ class ModelFormMixin(FormMixin, SingleObjectMixin):
     def get_success_url(self):
         """Return the URL to redirect to after processing a valid form."""
         if self.success_url:
-            url = self.success_url.format(**self.object.__dict__)
+            try:
+                url = self.success_url.format(**self.object.__dict__)
+            except KeyError as e:
+                raise ImproperlyConfigured(
+                    f"success_url '{self.success_url}' could not be formatted "
+                    f"with object attributes {list(self.object.__dict__.keys())}: "
+                    f"Missing key {e}"
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
+            except KeyError as e:
+                raise ImproperlyConfigured(
+                    f"success_url '{self.success_url}' could not be formatted "
+                    f"with object attributes {list(self.object.__dict__.keys())}: "
+                    f"Missing key {e}"
+                )
         else:
             raise ImproperlyConfigured("No URL to redirect to. Provide a success_url.")
```