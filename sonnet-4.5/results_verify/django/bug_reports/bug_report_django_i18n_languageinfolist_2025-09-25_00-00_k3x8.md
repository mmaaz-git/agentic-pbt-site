# Bug Report: django.templatetags.i18n.GetLanguageInfoListNode.get_language_info IndexError

**Target**: `django.templatetags.i18n.GetLanguageInfoListNode.get_language_info`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `GetLanguageInfoListNode.get_language_info` method crashes with an `IndexError` when processing empty language codes (empty strings or empty tuples/lists). The method accesses `language[0]` without first checking if `language` is empty.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.templatetags.i18n import GetLanguageInfoListNode

@given(st.one_of(
    st.just(''),
    st.just(()),
    st.just([]),
))
def test_get_language_info_should_not_crash_on_empty_input(language):
    node = GetLanguageInfoListNode(None, None)

    try:
        result = node.get_language_info(language)
    except IndexError:
        raise
```

**Failing inputs**:
- Empty string: `''`
- Empty tuple: `()`
- Empty list: `[]`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/path/to/django')

from django.conf import settings
settings.configure(INSTALLED_APPS=[], LANGUAGES=[('en', 'English')], USE_I18N=True)

from django.templatetags.i18n import GetLanguageInfoListNode

node = GetLanguageInfoListNode(None, None)

result = node.get_language_info('')
```

## Why This Is A Bug

The `{% get_language_info_list %}` template tag accepts a list of language codes or language tuples. According to the docstring, the input can be "a list of strings or a settings.LANGUAGES style list". When processing this list, the `get_language_info` method is called for each element.

If a user provides a list containing empty strings or empty sequences (which might happen due to data processing errors or edge cases), the method crashes with an `IndexError` instead of handling the error gracefully or providing a meaningful error message.

The code at `i18n.py:43` accesses `language[0]` without checking if `language` is empty. The method uses a heuristic to distinguish between:
- String language codes (e.g., 'en'): checks if `len(language[0]) > 1` is False
- Tuple/list language codes (e.g., ('en', 'English')): checks if `len(language[0]) > 1` is True

However, this heuristic fails when `language` is empty because `language[0]` raises `IndexError`.

## Fix

```diff
--- a/django/templatetags/i18n.py
+++ b/django/templatetags/i18n.py
@@ -39,9 +39,14 @@ class GetLanguageInfoListNode(Node):

     def get_language_info(self, language):
-        # ``language`` is either a language code string or a sequence
-        # with the language code as its first item
-        if len(language[0]) > 1:
-            return translation.get_language_info(language[0])
+        if not language:
+            raise ValueError("Language code cannot be empty")
+
+        # Check if language is a string or a sequence
+        if isinstance(language, str):
+            return translation.get_language_info(language)
         else:
-            return translation.get_language_info(str(language))
+            # language is a sequence with the language code as its first item
+            if not language[0]:
+                raise ValueError("Language code cannot be empty")
+            return translation.get_language_info(language[0])
```

This fix:
1. Checks for empty inputs before accessing `language[0]`
2. Uses `isinstance(language, str)` instead of the length heuristic, which is more robust
3. Provides clear error messages for invalid inputs