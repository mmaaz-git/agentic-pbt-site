# Bug Report: django.utils.translation.to_locale Incomplete Case Normalization for Non-Dashed Strings

**Target**: `django.utils.translation.to_locale`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `to_locale()` function only lowercases the first 3 characters of uppercase strings without dashes, leaving characters from position 4 onward in their original case, resulting in inconsistent normalization behavior.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.utils.translation import to_locale


@given(st.text(min_size=4, max_size=10, alphabet=st.characters(min_codepoint=65, max_codepoint=90)))
def test_to_locale_without_dash_should_be_lowercase(language_str):
    result = to_locale(language_str)
    assert result == result.lower(), f"to_locale({language_str!r}) = {result!r}, but should be all lowercase when no dash present"


if __name__ == "__main__":
    test_to_locale_without_dash_should_be_lowercase()
```

<details>

<summary>
**Failing input**: `'AAAA'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 12, in <module>
    test_to_locale_without_dash_should_be_lowercase()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 6, in test_to_locale_without_dash_should_be_lowercase
    def test_to_locale_without_dash_should_be_lowercase(language_str):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/10/hypo.py", line 8, in test_to_locale_without_dash_should_be_lowercase
    assert result == result.lower(), f"to_locale({language_str!r}) = {result!r}, but should be all lowercase when no dash present"
           ^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: to_locale('AAAA') = 'aaaA', but should be all lowercase when no dash present
Falsifying example: test_to_locale_without_dash_should_be_lowercase(
    language_str='AAAA',
)
```
</details>

## Reproducing the Bug

```python
from django.utils.translation import to_locale

# Test cases showing the bug
print("Testing to_locale() with uppercase strings without dashes:")
print(f"to_locale('AAAA') = {to_locale('AAAA')!r}")
print(f"to_locale('ENUS') = {to_locale('ENUS')!r}")
print(f"to_locale('FRCA') = {to_locale('FRCA')!r}")
print(f"to_locale('DEDE') = {to_locale('DEDE')!r}")
print(f"to_locale('ABCDEFGH') = {to_locale('ABCDEFGH')!r}")

print("\nFor comparison, strings with 3 or fewer characters:")
print(f"to_locale('EN') = {to_locale('EN')!r}")
print(f"to_locale('FRA') = {to_locale('FRA')!r}")

print("\nExpected behavior (what should happen):")
print("All characters should be lowercase when no dash is present")
print("Expected: 'aaaa', 'enus', 'frca', 'dede', 'abcdefgh', 'en', 'fra'")

print("\nActual behavior shows only first 3 chars are lowercased")
```

<details>

<summary>
Partial lowercasing - only first 3 characters are converted to lowercase
</summary>
```
Testing to_locale() with uppercase strings without dashes:
to_locale('AAAA') = 'aaaA'
to_locale('ENUS') = 'enuS'
to_locale('FRCA') = 'frcA'
to_locale('DEDE') = 'dedE'
to_locale('ABCDEFGH') = 'abcDEFGH'

For comparison, strings with 3 or fewer characters:
to_locale('EN') = 'en'
to_locale('FRA') = 'fra'

Expected behavior (what should happen):
All characters should be lowercase when no dash is present
Expected: 'aaaa', 'enus', 'frca', 'dede', 'abcdefgh', 'en', 'fra'

Actual behavior shows only first 3 chars are lowercased
```
</details>

## Why This Is A Bug

The function exhibits inconsistent normalization behavior that violates the principle of least surprise. When processing a string without a dash, the function attempts to normalize it to lowercase but fails to do so completely.

The bug occurs at line 235 in `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/utils/translation/__init__.py`:

```python
return language[:3].lower() + language[3:]
```

This line only lowercases the first 3 characters and concatenates the rest unchanged. However, line 233 has already computed the fully lowercased string in the variable `lang`:

```python
lang, _, country = language.lower().partition("-")
```

Since `country` is empty (no dash found), `lang` contains the entire lowercased string. The function should return `lang` instead of partially lowercasing the original input.

While the function's documentation focuses on converting hyphenated language codes (e.g., "en-us" to "en_US"), the implementation does attempt to handle non-dashed inputs. The partial lowercasing appears to be an oversight rather than intentional behavior, as there's no logical reason to lowercase only the first 3 characters.

## Relevant Context

The `to_locale()` function is part of Django's internationalization (i18n) infrastructure, primarily used to convert language codes from the hyphenated format (BCP 47/IETF standard like "en-us") to the POSIX locale format with underscores (like "en_US").

Standard language codes are typically 2-3 characters (ISO 639-1/2/3), which explains why the bug rarely manifests in production. However, the function's current behavior creates an inconsistency:
- Strings with ≤3 characters: fully lowercased (correct)
- Strings with >3 characters: only first 3 chars lowercased (buggy)

The companion function `to_language()` (lines 222-228) correctly handles the reverse conversion and consistently lowercases all characters when appropriate.

Django documentation: https://docs.djangoproject.com/en/stable/ref/utils/#django.utils.translation.to_locale
Source code: https://github.com/django/django/blob/main/django/utils/translation/__init__.py#L231-L244

## Proposed Fix

```diff
--- a/django/utils/translation/__init__.py
+++ b/django/utils/translation/__init__.py
@@ -232,7 +232,7 @@ def to_locale(language):
     """Turn a language name (en-us) into a locale name (en_US)."""
     lang, _, country = language.lower().partition("-")
     if not country:
-        return language[:3].lower() + language[3:]
+        return lang
     # A language with > 2 characters after the dash only has its first
     # character after the dash capitalized; e.g. sr-latn becomes sr_Latn.
     # A language with 2 characters after the dash has both characters
```