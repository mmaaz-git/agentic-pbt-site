# Bug Report: django.templatetags.i18n GetLanguageInfoListNode Incorrect Type Detection Heuristic

**Target**: `django.templatetags.i18n.GetLanguageInfoListNode.get_language_info`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_language_info()` method incorrectly distinguishes between string language codes and sequence inputs using a fragile character-length heuristic instead of proper type checking, causing it to fail on sequences with single-character first elements.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis property-based test for Django i18n GetLanguageInfoListNode bug"""

import os
import sys
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

import django
from django.conf import settings
settings.configure(
    DEBUG=True,
    LANGUAGE_CODE='en-us',
    USE_I18N=True,
    USE_L10N=True,
)
django.setup()

from hypothesis import given, strategies as st
from django.templatetags.i18n import GetLanguageInfoListNode

@given(st.lists(st.text(min_size=1, max_size=5), min_size=2, max_size=2))
def test_get_language_info_handles_sequences(language):
    node = GetLanguageInfoListNode(None, 'result')

    try:
        result = node.get_language_info(language)
    except TypeError as e:
        raise AssertionError(
            f"Should handle sequence {language} but got TypeError: {e}"
        )

if __name__ == "__main__":
    test_get_language_info_handles_sequences()
```

<details>

<summary>
**Failing input**: `['0', '0']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/utils/translation/__init__.py", line 273, in get_language_info
    lang_info = LANG_INFO[lang_code]
                ~~~~~~~~~^^^^^^^^^^^
KeyError: "['0', '0']"

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 33, in <module>
    test_get_language_info_handles_sequences()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 22, in test_get_language_info_handles_sequences
    def test_get_language_info_handles_sequences(language):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 26, in test_get_language_info_handles_sequences
    result = node.get_language_info(language)
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/templatetags/i18n.py", line 46, in get_language_info
    return translation.get_language_info(str(language))
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/utils/translation/__init__.py", line 280, in get_language_info
    raise KeyError("Unknown language code %s." % lang_code)
KeyError: "Unknown language code ['0', '0']."
Falsifying example: test_get_language_info_handles_sequences(
    language=['0', '0'],
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction case for Django i18n GetLanguageInfoListNode bug"""

import os
import sys
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

import django
from django.conf import settings
settings.configure(
    DEBUG=True,
    LANGUAGE_CODE='en-us',
    USE_I18N=True,
    USE_L10N=True,
)
django.setup()

from django.templatetags.i18n import GetLanguageInfoListNode
from django.utils import translation

# Create a node instance
node = GetLanguageInfoListNode(None, 'result')

print("=" * 60)
print("Testing GetLanguageInfoListNode.get_language_info()")
print("=" * 60)

# Test case 1: Sequence with single-character first element (FAILS)
print("\nTest 1: Sequence with single-character first element")
print("-" * 50)
language = ['x', 'Unknown Language']
print(f"Input: {language}")
print(f"Type: {type(language)}")
print(f"language[0] = {language[0]!r} (type: {type(language[0])})")
print(f"len(language[0]) = {len(language[0])}")

try:
    # Trace through the logic
    if len(language[0]) > 1:
        print("Logic: len(language[0]) > 1 is TRUE")
        print(f"Would call: translation.get_language_info({language[0]!r})")
    else:
        print("Logic: len(language[0]) > 1 is FALSE")
        print(f"Would call: translation.get_language_info(str({language!r}))")
        print(f"         = translation.get_language_info({str(language)!r})")

    # Actually call the method
    result = node.get_language_info(language)
    print(f"SUCCESS: Result = {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# Test case 2: Sequence with multi-character first element (WORKS)
print("\n" + "=" * 60)
print("Test 2: Sequence with multi-character first element")
print("-" * 50)
language = ['en', 'English']
print(f"Input: {language}")
print(f"Type: {type(language)}")
print(f"language[0] = {language[0]!r} (type: {type(language[0])})")
print(f"len(language[0]) = {len(language[0])}")

try:
    # Trace through the logic
    if len(language[0]) > 1:
        print("Logic: len(language[0]) > 1 is TRUE")
        print(f"Would call: translation.get_language_info({language[0]!r})")
    else:
        print("Logic: len(language[0]) > 1 is FALSE")
        print(f"Would call: translation.get_language_info(str({language!r}))")
        print(f"         = translation.get_language_info({str(language)!r})")

    # Actually call the method
    result = node.get_language_info(language)
    print(f"SUCCESS: Result = {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# Test case 3: String language code (WORKS by coincidence)
print("\n" + "=" * 60)
print("Test 3: String language code")
print("-" * 50)
language = 'en'
print(f"Input: {language!r}")
print(f"Type: {type(language)}")
print(f"language[0] = {language[0]!r} (type: {type(language[0])})")
print(f"len(language[0]) = {len(language[0])}")

try:
    # Trace through the logic
    if len(language[0]) > 1:
        print("Logic: len(language[0]) > 1 is TRUE")
        print(f"Would call: translation.get_language_info({language[0]!r})")
    else:
        print("Logic: len(language[0]) > 1 is FALSE")
        print(f"Would call: translation.get_language_info(str({language!r}))")
        print(f"         = translation.get_language_info({str(language)!r})")

    # Actually call the method
    result = node.get_language_info(language)
    print(f"SUCCESS: Result = {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)
print("""
The bug occurs because the code uses len(language[0]) > 1 to distinguish
between strings and sequences. This fails for sequences with single-character
first elements:

- For ['x', 'Unknown']: language[0] = 'x', len('x') = 1, so it calls
  translation.get_language_info(str(['x', 'Unknown'])) which is
  translation.get_language_info("['x', 'Unknown']") - INVALID!

- For ['en', 'English']: language[0] = 'en', len('en') = 2, so it calls
  translation.get_language_info('en') - CORRECT!

- For 'en': language[0] = 'e', len('e') = 1, so it calls
  translation.get_language_info(str('en')) which is
  translation.get_language_info('en') - WORKS BY COINCIDENCE!

The fix is to use isinstance(language, str) to properly check the type.
""")
```

<details>

<summary>
KeyError when processing sequence with single-character first element
</summary>
```
============================================================
Testing GetLanguageInfoListNode.get_language_info()
============================================================

Test 1: Sequence with single-character first element
--------------------------------------------------
Input: ['x', 'Unknown Language']
Type: <class 'list'>
language[0] = 'x' (type: <class 'str'>)
len(language[0]) = 1
Logic: len(language[0]) > 1 is FALSE
Would call: translation.get_language_info(str(['x', 'Unknown Language']))
         = translation.get_language_info("['x', 'Unknown Language']")
ERROR: KeyError: "Unknown language code ['x', 'Unknown Language']."

============================================================
Test 2: Sequence with multi-character first element
--------------------------------------------------
Input: ['en', 'English']
Type: <class 'list'>
language[0] = 'en' (type: <class 'str'>)
len(language[0]) = 2
Logic: len(language[0]) > 1 is TRUE
Would call: translation.get_language_info('en')
SUCCESS: Result = {'bidi': False, 'code': 'en', 'name': 'English', 'name_local': 'English', 'name_translated': 'English'}

============================================================
Test 3: String language code
--------------------------------------------------
Input: 'en'
Type: <class 'str'>
language[0] = 'e' (type: <class 'str'>)
len(language[0]) = 1
Logic: len(language[0]) > 1 is FALSE
Would call: translation.get_language_info(str('en'))
         = translation.get_language_info('en')
SUCCESS: Result = {'bidi': False, 'code': 'en', 'name': 'English', 'name_local': 'English', 'name_translated': 'English'}

============================================================
ANALYSIS
============================================================

The bug occurs because the code uses len(language[0]) > 1 to distinguish
between strings and sequences. This fails for sequences with single-character
first elements:

- For ['x', 'Unknown']: language[0] = 'x', len('x') = 1, so it calls
  translation.get_language_info(str(['x', 'Unknown'])) which is
  translation.get_language_info("['x', 'Unknown']") - INVALID!

- For ['en', 'English']: language[0] = 'en', len('en') = 2, so it calls
  translation.get_language_info('en') - CORRECT!

- For 'en': language[0] = 'e', len('e') = 1, so it calls
  translation.get_language_info(str('en')) which is
  translation.get_language_info('en') - WORKS BY COINCIDENCE!

The fix is to use isinstance(language, str) to properly check the type.

```
</details>

## Why This Is A Bug

The `get_language_info()` method at lines 40-46 in `/django/templatetags/i18n.py` violates its own documented behavior. The function's docstring (lines 268-271) and inline comment (lines 41-42) explicitly state:

> "Store a list of language information dictionaries for the given language codes. The language codes can be specified either as a list of strings or a settings.LANGUAGES style list (or any sequence of sequences whose first items are language codes)."

The comment at line 41-42 confirms: "``language`` is either a language code string or a sequence with the language code as its first item"

However, the implementation uses a fragile heuristic `len(language[0]) > 1` to distinguish between strings and sequences, which:

1. **Fails on valid inputs**: Any sequence with a single-character first element (e.g., `['x', 'Custom']`, `['0', '0']`) causes a KeyError because the code incorrectly converts the entire list to a string representation and passes `"['x', 'Custom']"` to `translation.get_language_info()`.

2. **Relies on coincidence**: String inputs like `'en'` only work because `str('en')` equals `'en'`. The logic path is wrong but produces the correct result by accident.

3. **Violates type checking principles**: The code attempts to infer type from data characteristics rather than using Python's `isinstance()` for proper type checking.

4. **Contradicts documentation**: The documentation promises support for "any sequence of sequences whose first items are language codes" without restricting the length of those codes.

## Relevant Context

The bug exists in Django's template tag system, specifically in the `{% get_language_info_list %}` tag implementation. This tag is used in Django templates to retrieve language information for display purposes. While most standard language codes are 2+ characters (like 'en', 'es', 'fr'), the bug affects:

- Custom single-character language identifiers
- Test data with simplified codes
- Any programmatically generated sequences that might have single-character elements

The actual Django source code location: `/django/templatetags/i18n.py`, lines 40-46

Django documentation reference: https://docs.djangoproject.com/en/stable/topics/i18n/translation/#get-language-info-list

## Proposed Fix

```diff
--- a/django/templatetags/i18n.py
+++ b/django/templatetags/i18n.py
@@ -40,10 +40,11 @@ class GetLanguageInfoListNode(Node):
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

     def render(self, context):
         langs = self.languages.resolve(context)
```