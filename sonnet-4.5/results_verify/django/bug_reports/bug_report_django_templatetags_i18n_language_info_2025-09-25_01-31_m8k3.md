# Bug Report: django.templatetags.i18n GetLanguageInfoListNode.get_language_info Type Handling

**Target**: `django.templatetags.i18n.GetLanguageInfoListNode.get_language_info`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_language_info()` method uses a fragile heuristic to distinguish between string language codes and sequence inputs. It checks `len(language[0]) > 1` which fails for sequences with single-character first elements, and relies on string indexing behavior rather than proper type checking.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(st.lists(st.text(min_size=1, max_size=5), min_size=2, max_size=2))
def test_get_language_info_handles_sequences(language):
    node = GetLanguageInfoListNode(None, 'result')

    try:
        result = node.get_language_info(language)
    except TypeError as e:
        raise AssertionError(
            f"Should handle sequence {language} but got TypeError: {e}"
        )
```

**Failing input**: `['x', 'Unknown Language']` (a sequence with single-char first element)

## Reproducing the Bug

```python
from django.templatetags.i18n import GetLanguageInfoListNode

node = GetLanguageInfoListNode(None, 'result')

language = ['x', 'Unknown Language']

print(f"Input: {language}")
print(f"language[0] = {language[0]!r}")
print(f"len(language[0]) = {len(language[0])}")

if len(language[0]) > 1:
    print("Branch: if - would call translation.get_language_info(language[0])")
else:
    print("Branch: else - calls str(language)")
    print(f"str(language) = {str(language)!r}")
    print("This will fail because it tries to look up the string representation!")
```

Output:
```
Input: ['x', 'Unknown Language']
language[0] = 'x'
len(language[0]) = 1
Branch: else - calls str(language)
str(language) = "['x', 'Unknown Language']"
This will fail because it tries to look up the string representation!
```

## Why This Is A Bug

The code at lines 43-46 in `i18n.py`:

```python
if len(language[0]) > 1:
    return translation.get_language_info(language[0])
else:
    return translation.get_language_info(str(language))
```

Uses a fragile heuristic based on the assumption that:
- Strings have single-character first elements: `"en"[0]` is `"e"`, length 1
- Sequences have multi-character first elements: `["en", "English"]` has `"en"` at index 0, length 2+

This breaks for:
1. **Sequences with single-character codes**: `["x", "Unknown"]` → `language[0]` is `"x"`, length 1 → falls to else branch → calls `str(["x", "Unknown"])` which is `"['x', 'Unknown']"` ✗

2. **Confusing logic**: The check `len(language[0])` is checking the length of the first element, not distinguishing types.

The docstring says "language is either a language code string or a sequence", so the code should check the type, not rely on character length.

## Fix

```diff
--- a/django/templatetags/i18n.py
+++ b/django/templatetags/i18n.py
@@ -40,10 +40,12 @@ class GetLanguageInfoListNode(Node):
     def get_language_info(self, language):
         # ``language`` is either a language code string or a sequence
         # with the language code as its first item
-        if len(language[0]) > 1:
-            return translation.get_language_info(language[0])
-        else:
+        if isinstance(language, str):
             return translation.get_language_info(str(language))
+        else:
+            # language is a sequence with language code as first item
+            return translation.get_language_info(language[0])
+
     def render(self, context):
         langs = self.languages.resolve(context)
```