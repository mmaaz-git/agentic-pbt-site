# Bug Report: GetLanguageInfoListNode.get_language_info Single-Character Language Code

**Target**: `django.templatetags.i18n.GetLanguageInfoListNode.get_language_info`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_language_info` method incorrectly handles tuples with single-character language codes. It uses `len(language[0]) > 1` to distinguish between strings and tuples, but this fails when a tuple contains a single-character code (e.g., `('x', 'Language X')`), causing it to call `str(language)` which produces `"('x', 'Language X')"` instead of extracting the code `'x'`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.templatetags.i18n import GetLanguageInfoListNode

@given(
    lang_code=st.text(min_size=1, max_size=1, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
    lang_name=st.text(min_size=1, max_size=20)
)
def test_single_char_tuple(lang_code, lang_name):
    node = GetLanguageInfoListNode(None, 'test_var')
    language = (lang_code, lang_name)
    result = node.get_language_info(language)
```

**Failing input**: `lang_code='a', lang_name='-'` (resulting in `language = ('a', '-')`)

## Reproducing the Bug

```python
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from django.conf import settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        USE_I18N=True,
        LANGUAGES=[],
        LANGUAGE_CODE='en',
    )

import django
django.setup()

from django.templatetags.i18n import GetLanguageInfoListNode

node = GetLanguageInfoListNode(None, 'test_var')

print("Works correctly with two-char code:")
node.get_language_info(("en", "English"))

print("Fails with single-char code:")
node.get_language_info(("x", "Language X"))
```

## Why This Is A Bug

The method's docstring states: "language is either a language code string or a sequence with the language code as its first item."

The current logic uses `len(language[0]) > 1` to differentiate:
- For strings like `"en"`, `language[0]` is `'e'` (length 1), so it correctly uses `str(language)` = `"en"`
- For tuples like `("en", "English")`, `language[0]` is `"en"` (length 2), so it correctly uses `language[0]` = `"en"`
- For tuples like `("x", "Language X")`, `language[0]` is `"x"` (length 1), so it incorrectly uses `str(language)` = `"('x', 'Language X')"`

This causes the function to look up the string representation of the tuple as a language code, which will always fail.

## Fix

```diff
--- a/django/templatetags/i18n.py
+++ b/django/templatetags/i18n.py
@@ -40,10 +40,10 @@ class GetLanguageInfoListNode(Node):
     def get_language_info(self, language):
         # ``language`` is either a language code string or a sequence
         # with the language code as its first item
-        if len(language[0]) > 1:
-            return translation.get_language_info(language[0])
-        else:
+        if isinstance(language, str):
             return translation.get_language_info(str(language))
+        else:
+            return translation.get_language_info(language[0])
```