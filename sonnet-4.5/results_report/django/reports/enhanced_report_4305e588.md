# Bug Report: GetLanguageInfoListNode.get_language_info Incorrect Type Detection for Single-Character Language Codes

**Target**: `django.templatetags.i18n.GetLanguageInfoListNode.get_language_info`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_language_info` method incorrectly distinguishes between string and tuple inputs using `len(language[0]) > 1`, causing it to pass the string representation of tuples containing single-character language codes (e.g., `"('x', 'Language X')"`) to the translation lookup instead of extracting just the language code (`'x'`).

## Property-Based Test

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

from hypothesis import given, strategies as st
from django.templatetags.i18n import GetLanguageInfoListNode

@given(
    lang_code=st.text(min_size=1, max_size=1, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
    lang_name=st.text(min_size=1, max_size=20)
)
def test_single_char_tuple(lang_code, lang_name):
    node = GetLanguageInfoListNode(None, 'test_var')
    language = (lang_code, lang_name)

    # The function should extract the first element of the tuple
    # But due to the bug, it will pass str(language) instead
    try:
        result = node.get_language_info(language)
        # If it succeeds, check that the code matches
        assert result.get('code') == lang_code, f"Expected code {lang_code}, got {result.get('code')}"
    except KeyError as e:
        # The error message will contain the string representation of the tuple
        # instead of just the language code
        error_msg = str(e)
        if str(language) in error_msg:
            print(f"BUG CONFIRMED: Error message contains tuple string representation")
            print(f"  Input tuple: {language}")
            print(f"  Error message: {error_msg}")
            print(f"  Expected to look up: '{lang_code}'")
            print(f"  Actually looked up: '{str(language)}'")
            raise AssertionError(f"Bug detected: function passed '{str(language)}' instead of '{lang_code}'")
        raise

# Run the test
if __name__ == "__main__":
    print("Running property-based test to find the bug...")
    print("=" * 60)
    try:
        test_single_char_tuple()
        print("No failures found after running multiple test cases")
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        print("\nThis demonstrates the bug in GetLanguageInfoListNode.get_language_info()")
        print("The method incorrectly handles tuples with single-character language codes.")
```

<details>

<summary>
**Failing input**: `lang_code='a', lang_name='0'`
</summary>
```
Running property-based test to find the bug...
============================================================
BUG CONFIRMED: Error message contains tuple string representation
  Input tuple: ('a', '0')
  Error message: "Unknown language code ('a', '0')."
  Expected to look up: 'a'
  Actually looked up: '('a', '0')'
[... multiple similar confirmations truncated for brevity ...]

✗ Test failed: Bug detected: function passed '('a', '0')' instead of 'a'

This demonstrates the bug in GetLanguageInfoListNode.get_language_info()
The method incorrectly handles tuples with single-character language codes.
```
</details>

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
        LANGUAGES=[('x', 'Language X')],  # Custom single-character language code
        LANGUAGE_CODE='en',
    )

import django
django.setup()

from django.templatetags.i18n import GetLanguageInfoListNode

# Create an instance of GetLanguageInfoListNode
node = GetLanguageInfoListNode(None, 'test_var')

# Test case 1: Works correctly with two-character code
print("Test 1: Two-character language code in tuple")
print("Input: ('en', 'English')")
try:
    result = node.get_language_info(('en', 'English'))
    print(f"Result: {result}")
    print(f"Success - extracted language code correctly\n")
except Exception as e:
    print(f"Error: {e}\n")

# Test case 2: Fails with single-character code
print("Test 2: Single-character language code in tuple")
print("Input: ('x', 'Language X')")
try:
    result = node.get_language_info(('x', 'Language X'))
    print(f"Result: {result}")
    print(f"Success - extracted language code correctly\n")
except Exception as e:
    print(f"Error: {e}\n")

# Test case 3: String input with two-character code
print("Test 3: String input with two-character code")
print("Input: 'en'")
try:
    result = node.get_language_info('en')
    print(f"Result: {result}")
    print(f"Success - processed string correctly\n")
except Exception as e:
    print(f"Error: {e}\n")

# Test case 4: String input with single-character code (hypothetical)
print("Test 4: String input with single-character (hypothetical)")
print("Input: 'x'")
try:
    result = node.get_language_info('x')
    print(f"Result: {result}")
    print(f"Success - processed string correctly\n")
except Exception as e:
    print(f"Error: {e}\n")

# Demonstrate the bug: what actually gets passed to get_language_info
print("Debug: What gets passed to translation.get_language_info()")
print("------------------------------------------------------")
test_tuple = ('x', 'Language X')
print(f"For tuple {test_tuple}:")
print(f"  language[0] = '{test_tuple[0]}' (length = {len(test_tuple[0])})")
print(f"  len(language[0]) > 1 evaluates to: {len(test_tuple[0]) > 1}")
if len(test_tuple[0]) > 1:
    print(f"  Would pass: language[0] = '{test_tuple[0]}'")
else:
    print(f"  Would pass: str(language) = '{str(test_tuple)}'")
print(f"  The string '{str(test_tuple)}' is NOT a valid language code!")
```

<details>

<summary>
Reproduction output showing the bug
</summary>
```
Test 1: Two-character language code in tuple
Input: ('en', 'English')
Result: {'bidi': False, 'code': 'en', 'name': 'English', 'name_local': 'English', 'name_translated': 'English'}
Success - extracted language code correctly

Test 2: Single-character language code in tuple
Input: ('x', 'Language X')
Error: "Unknown language code ('x', 'Language X')."

Test 3: String input with two-character code
Input: 'en'
Result: {'bidi': False, 'code': 'en', 'name': 'English', 'name_local': 'English', 'name_translated': 'English'}
Success - processed string correctly

Test 4: String input with single-character (hypothetical)
Input: 'x'
Error: 'Unknown language code x.'

Debug: What gets passed to translation.get_language_info()
------------------------------------------------------
For tuple ('x', 'Language X'):
  language[0] = 'x' (length = 1)
  len(language[0]) > 1 evaluates to: False
  Would pass: str(language) = '('x', 'Language X')'
  The string '('x', 'Language X')' is NOT a valid language code!
```
</details>

## Why This Is A Bug

The method's documentation clearly states: "language is either a language code string or a sequence with the language code as its first item" (lines 41-42 in `/django/templatetags/i18n.py`).

The current implementation uses `len(language[0]) > 1` as a heuristic to distinguish between strings and tuples/sequences. This logic is fundamentally flawed:

1. **For strings like `"en"`**: `language[0]` returns `'e'` (the first character), which has length 1. The condition `len(language[0]) > 1` is False, so it correctly calls `str(language)` which returns `"en"`.

2. **For tuples like `("en", "English")`**: `language[0]` returns `"en"` (the first element), which has length 2. The condition `len(language[0]) > 1` is True, so it correctly extracts `language[0]` = `"en"`.

3. **For tuples like `("x", "Language X")`**: `language[0]` returns `"x"` (the first element), which has length 1. The condition `len(language[0]) > 1` is False, so it incorrectly calls `str(language)`, which returns `"('x', 'Language X')"` - the string representation of the entire tuple.

This causes the function to look up `"('x', 'Language X')"` as a language code in Django's language registry, which will always fail with a KeyError.

## Relevant Context

This bug affects the `get_language_info_list` template tag when used with sequences containing single-character language codes. While Django's built-in `LANG_INFO` dictionary (in `/django/conf/locale/__init__.py`) only contains 2+ character language codes, users can define custom language codes in their Django settings using the `LANGUAGES` setting.

The template tag documentation (lines 267-271) explicitly states it accepts "a settings.LANGUAGES style list (or any sequence of sequences whose first items are language codes)" without any restriction on language code length.

Related Django documentation:
- Template tag: https://docs.djangoproject.com/en/stable/topics/i18n/translation/#get-language-info-list
- Language codes: https://docs.djangoproject.com/en/stable/ref/settings/#languages

## Proposed Fix

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