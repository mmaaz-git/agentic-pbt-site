# Bug Report: django.templatetags.i18n GetLanguageInfoListNode IndexError on Empty Input

**Target**: `django.templatetags.i18n.GetLanguageInfoListNode.get_language_info`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `GetLanguageInfoListNode.get_language_info` method crashes with an `IndexError` when passed empty strings or empty sequences, instead of raising the more appropriate `KeyError` that would indicate an invalid language code.

## Property-Based Test

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        LANGUAGES=[('en', 'English'), ('fr', 'French'), ('de', 'German')],
        USE_I18N=True,
        USE_L10N=True,
        SECRET_KEY='test-secret-key',
        INSTALLED_APPS=[],
    )
    django.setup()

from hypothesis import given, strategies as st
from django.templatetags.i18n import GetLanguageInfoListNode


@given(st.one_of(
    st.text(min_size=0, max_size=10),
    st.lists(st.text(), min_size=0, max_size=5),
    st.tuples(st.text(min_size=0, max_size=5), st.text()),
))
def test_get_language_info_with_arbitrary_input(language):
    node = GetLanguageInfoListNode(None, 'result')
    result = node.get_language_info(language)
    assert result is not None

# Run the test
if __name__ == "__main__":
    test_get_language_info_with_arbitrary_input()
```

<details>

<summary>
**Failing input**: `language=''`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 31, in <module>
  |     test_get_language_info_with_arbitrary_input()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 20, in test_get_language_info_with_arbitrary_input
  |     st.text(min_size=0, max_size=10),
  |                ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/utils/translation/__init__.py", line 273, in get_language_info
    |     lang_info = LANG_INFO[lang_code]
    |                 ~~~~~~~~~^^^^^^^^^^^
    | KeyError: '0'
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 26, in test_get_language_info_with_arbitrary_input
    |     result = node.get_language_info(language)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/templatetags/i18n.py", line 46, in get_language_info
    |     return translation.get_language_info(str(language))
    |            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/utils/translation/__init__.py", line 280, in get_language_info
    |     raise KeyError("Unknown language code %s." % lang_code)
    | KeyError: 'Unknown language code 0.'
    | Falsifying example: test_get_language_info_with_arbitrary_input(
    |     language='0',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 26, in test_get_language_info_with_arbitrary_input
    |     result = node.get_language_info(language)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/templatetags/i18n.py", line 43, in get_language_info
    |     if len(language[0]) > 1:
    |            ~~~~~~~~^^^
    | IndexError: string index out of range
    | Falsifying example: test_get_language_info_with_arbitrary_input(
    |     language='',
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        LANGUAGES=[('en', 'English'), ('fr', 'French'), ('de', 'German')],
        USE_I18N=True,
        USE_L10N=True,
        SECRET_KEY='test-secret-key',
        INSTALLED_APPS=[],
    )
    django.setup()

from django.templatetags.i18n import GetLanguageInfoListNode

node = GetLanguageInfoListNode(None, 'result')

# Test with empty string
print("Testing with empty string '':")
try:
    result = node.get_language_info('')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

# Test with empty list
print("\nTesting with empty list []:")
try:
    result = node.get_language_info([])
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

# Test with empty tuple
print("\nTesting with empty tuple ():")
try:
    result = node.get_language_info(())
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

# For comparison - test with valid language code
print("\nTesting with valid language code 'en':")
try:
    result = node.get_language_info('en')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

# Test with invalid but non-empty language code
print("\nTesting with invalid language code 'xyz':")
try:
    result = node.get_language_info('xyz')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")
```

<details>

<summary>
IndexError crashes on empty inputs
</summary>
```
Testing with empty string '':
Exception: IndexError: string index out of range

Testing with empty list []:
Exception: IndexError: list index out of range

Testing with empty tuple ():
Exception: IndexError: tuple index out of range

Testing with valid language code 'en':
Result: {'bidi': False, 'code': 'en', 'name': 'English', 'name_local': 'English', 'name_translated': 'English'}

Testing with invalid language code 'xyz':
Exception: KeyError: 'Unknown language code xyz.'
```
</details>

## Why This Is A Bug

This violates expected behavior in several ways:

1. **Inconsistent error handling**: The function raises different exception types for the same category of problem (invalid language codes). Non-empty invalid codes like 'xyz' raise `KeyError: 'Unknown language code xyz.'` while empty strings raise `IndexError: string index out of range`.

2. **Unhelpful error messages**: The `IndexError` provides no context about what went wrong. Users see "string index out of range" instead of "Unknown language code" which would indicate the actual problem.

3. **Violates function contract**: The code comment at line 41-42 states "language is either a language code string or a sequence with the language code as its first item". The function attempts to handle both cases but fails to check if the input is empty before accessing `language[0]`.

4. **Defensive programming failure**: The underlying `translation.get_language_info()` function properly handles invalid codes with a descriptive KeyError. The wrapper function defeats this error handling by crashing before the validation can occur.

## Relevant Context

The bug occurs in `django/templatetags/i18n.py` at line 43 in the `GetLanguageInfoListNode.get_language_info` method:

```python
def get_language_info(self, language):
    # ``language`` is either a language code string or a sequence
    # with the language code as its first item
    if len(language[0]) > 1:  # Line 43 - crashes here on empty input
        return translation.get_language_info(language[0])
    else:
        return translation.get_language_info(str(language))
```

This function is used by the Django template tag `{% get_language_info_list %}` to retrieve language information for multiple language codes. In normal template usage, this would receive data from template parsing which would not produce empty values, making this a rare edge case. However, when the function is used programmatically or in testing scenarios, empty inputs can occur.

The underlying `django.utils.translation.get_language_info()` function at line 269-287 properly validates language codes and raises `KeyError("Unknown language code %s." % lang_code)` for invalid codes, including empty strings when passed directly.

## Proposed Fix

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