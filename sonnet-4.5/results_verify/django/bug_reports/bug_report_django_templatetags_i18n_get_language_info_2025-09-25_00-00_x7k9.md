# Bug Report: django.templatetags.i18n GetLanguageInfoListNode IndexError on Empty Input

**Target**: `django.templatetags.i18n.GetLanguageInfoListNode.get_language_info`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `GetLanguageInfoListNode.get_language_info` method crashes with an `IndexError` when passed an empty string or empty sequence, rather than handling these edge cases gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.templatetags.i18n import GetLanguageInfoListNode
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        LANGUAGES=[('en', 'English'), ('fr', 'French'), ('de', 'German')],
    )


@given(st.one_of(
    st.text(min_size=0, max_size=10),
    st.lists(st.text(), min_size=0, max_size=5),
    st.tuples(st.text(min_size=0, max_size=5), st.text()),
))
def test_get_language_info_with_arbitrary_input(language):
    node = GetLanguageInfoListNode(None, 'result')
    result = node.get_language_info(language)
    assert result is not None
```

**Failing input**: `language=''`

## Reproducing the Bug

```python
from django.templatetags.i18n import GetLanguageInfoListNode
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        LANGUAGES=[('en', 'English'), ('fr', 'French')],
    )

node = GetLanguageInfoListNode(None, 'result')

node.get_language_info('')
```

This raises:
```
IndexError: string index out of range
```

The same bug occurs with empty tuples `()` and empty lists `[]`.

## Why This Is A Bug

The function's docstring states it handles "a list of strings or a settings.LANGUAGES style list" without specifying that strings must be non-empty. The code attempts to access `language[0]` without first checking if the sequence is empty, causing an IndexError rather than a more informative error message.

While empty language codes are invalid, the function should fail gracefully with a clear error message (e.g., `KeyError: 'Unknown language code.'`) rather than crashing with an IndexError that doesn't indicate the actual problem.

## Fix

```diff
--- a/django/templatetags/i18n.py
+++ b/django/templatetags/i18n.py
@@ -40,7 +40,7 @@ class GetLanguageInfoListNode(Node):
     def get_language_info(self, language):
         # ``language`` is either a language code string or a sequence
         # with the language code as its first item
-        if len(language[0]) > 1:
+        if language and len(language[0]) > 1:
             return translation.get_language_info(language[0])
         else:
             return translation.get_language_info(str(language))
```

This fix checks if `language` is non-empty before attempting to access `language[0]`, allowing empty strings and sequences to fall through to the `else` branch where they will be handled by `translation.get_language_info(str(language))`, which will raise a more appropriate `KeyError: 'Unknown language code.'`.