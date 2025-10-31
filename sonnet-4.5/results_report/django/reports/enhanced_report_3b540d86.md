# Bug Report: django.templatetags.i18n.GetLanguageInfoListNode.get_language_info IndexError on Empty Input

**Target**: `django.templatetags.i18n.GetLanguageInfoListNode.get_language_info`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_language_info` method in Django's i18n template tags crashes with an unhelpful `IndexError` when processing empty inputs (empty strings, tuples, or lists), instead of providing a meaningful error message.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test demonstrating the IndexError bug in
django.templatetags.i18n.GetLanguageInfoListNode.get_language_info
"""

import sys

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Configure Django settings
from django.conf import settings
settings.configure(
    INSTALLED_APPS=['django.contrib.contenttypes'],
    LANGUAGES=[('en', 'English'), ('es', 'Spanish')],
    USE_I18N=True,
    USE_TZ=False,
    SECRET_KEY='test-secret-key'
)

# Initialize Django
import django
django.setup()

from hypothesis import given, strategies as st
from django.templatetags.i18n import GetLanguageInfoListNode

@given(st.one_of(
    st.just(''),
    st.just(()),
    st.just([]),
))
def test_get_language_info_should_not_crash_on_empty_input(language):
    """Test that get_language_info handles empty inputs gracefully"""
    node = GetLanguageInfoListNode(None, None)

    try:
        result = node.get_language_info(language)
        # If we get here without crashing, that's okay
    except IndexError:
        # This is the bug - we're catching IndexError and re-raising it
        # to make the test fail and demonstrate the bug
        raise

# Run the test
if __name__ == "__main__":
    test_get_language_info_should_not_crash_on_empty_input()
```

<details>

<summary>
**Failing input**: `language=''`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 48, in <module>
    test_get_language_info_should_not_crash_on_empty_input()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 30, in test_get_language_info_should_not_crash_on_empty_input
    st.just(''),
               ^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 39, in test_get_language_info_should_not_crash_on_empty_input
    result = node.get_language_info(language)
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/templatetags/i18n.py", line 43, in get_language_info
    if len(language[0]) > 1:
           ~~~~~~~~^^^
IndexError: string index out of range
Falsifying example: test_get_language_info_should_not_crash_on_empty_input(
    language='',
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of the IndexError bug in django.templatetags.i18n.GetLanguageInfoListNode.get_language_info
"""

import sys

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Configure Django settings
from django.conf import settings
settings.configure(
    INSTALLED_APPS=['django.contrib.contenttypes'],
    LANGUAGES=[('en', 'English'), ('es', 'Spanish')],
    USE_I18N=True,
    USE_TZ=False,
    SECRET_KEY='test-secret-key'
)

# Initialize Django
import django
django.setup()

# Import the class with the bug
from django.templatetags.i18n import GetLanguageInfoListNode

# Create an instance
node = GetLanguageInfoListNode(None, None)

# Test with empty string - This will crash with IndexError
print("Testing with empty string '':")
try:
    result = node.get_language_info('')
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError: {e}")

# Test with empty tuple - This will also crash
print("\nTesting with empty tuple ():")
try:
    result = node.get_language_info(())
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError: {e}")

# Test with empty list - This will also crash
print("\nTesting with empty list []:")
try:
    result = node.get_language_info([])
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError: {e}")

# Test with valid input for comparison
print("\nTesting with valid input 'en':")
try:
    result = node.get_language_info('en')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
```

<details>

<summary>
IndexError crashes on empty inputs
</summary>
```
Testing with empty string '':
IndexError: string index out of range

Testing with empty tuple ():
IndexError: tuple index out of range

Testing with empty list []:
IndexError: list index out of range

Testing with valid input 'en':
Result: {'bidi': False, 'code': 'en', 'name': 'English', 'name_local': 'English', 'name_translated': 'English'}
```
</details>

## Why This Is A Bug

This violates expected behavior in several ways:

1. **Unhelpful Error Messages**: The code crashes with generic `IndexError` messages like "string index out of range" which provide no context about what went wrong. Django typically provides informative error messages for invalid inputs.

2. **Inconsistent Error Handling**: The underlying `translation.get_language_info()` function handles empty strings more gracefully by raising a `KeyError` with the message "Unknown language code ." which is much more informative than an `IndexError`.

3. **Documentation Gap**: The Django documentation for `{% get_language_info_list %}` states it accepts "a list of language codes" or "a settings.LANGUAGES style list" but doesn't specify how empty values should be handled. Users might reasonably expect either graceful handling or a clear error message.

4. **Code Quality Issue**: The method accesses `language[0]` at line 43 without first checking if the input is empty. This is a basic defensive programming oversight - the code should validate inputs before accessing array indices.

5. **Real-World Scenarios**: Empty values can occur in production through data processing errors, filtering operations that leave empty strings, or user input that isn't properly validated. When this happens, developers need clear error messages to debug the issue.

## Relevant Context

The bug is located in `/django/templatetags/i18n.py` at line 43 in the `get_language_info` method:

```python
def get_language_info(self, language):
    # ``language`` is either a language code string or a sequence
    # with the language code as its first item
    if len(language[0]) > 1:  # Line 43 - crashes here on empty input
        return translation.get_language_info(language[0])
    else:
        return translation.get_language_info(str(language))
```

The method tries to use a heuristic to distinguish between:
- String language codes (e.g., 'en')
- Tuple/list language codes (e.g., ('en', 'English'))

However, this heuristic fails when the input is empty because accessing `language[0]` raises an `IndexError` before any validation can occur.

Django documentation: https://docs.djangoproject.com/en/stable/topics/i18n/translation/#get-language-info-list
Source code: https://github.com/django/django/blob/main/django/templatetags/i18n.py

## Proposed Fix

```diff
--- a/django/templatetags/i18n.py
+++ b/django/templatetags/i18n.py
@@ -40,9 +40,16 @@ class GetLanguageInfoListNode(Node):
     def get_language_info(self, language):
         # ``language`` is either a language code string or a sequence
         # with the language code as its first item
-        if len(language[0]) > 1:
-            return translation.get_language_info(language[0])
+        if not language:
+            # Provide a clear error message for empty inputs
+            raise ValueError(f"Empty language code provided: {language!r}")
+
+        # Check if it's a string or a sequence
+        if isinstance(language, str):
+            return translation.get_language_info(language)
         else:
-            return translation.get_language_info(str(language))
+            # It's a sequence - get the first element (language code)
+            if not language[0]:
+                raise ValueError(f"Empty language code in sequence: {language!r}")
+            return translation.get_language_info(language[0])
```