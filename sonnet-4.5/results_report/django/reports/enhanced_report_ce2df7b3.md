# Bug Report: django.utils.translation.to_locale Inconsistent Case Handling

**Target**: `django.utils.translation.to_locale`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `to_locale()` function incorrectly handles case normalization for language codes without hyphens, resulting in mixed-case output like `'gerMAN'` from input `'GerMAN'` due to using the original input variable instead of the already-lowercased variable.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for django.utils.translation.to_locale bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, example
from django.utils.translation import to_locale


@given(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz', min_size=4, max_size=10))
@settings(max_examples=1000)
@example('GerMAN')
def test_to_locale_case_handling_without_dash(language):
    locale = to_locale(language)
    assert locale.islower(), (
        f"to_locale('{language}') = '{locale}' has inconsistent casing"
    )

if __name__ == "__main__":
    test_to_locale_case_handling_without_dash()
```

<details>

<summary>
**Failing input**: `'GerMAN'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 21, in <module>
    test_to_locale_case_handling_without_dash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 12, in test_to_locale_case_handling_without_dash
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 16, in test_to_locale_case_handling_without_dash
    assert locale.islower(), (
           ~~~~~~~~~~~~~~^^
AssertionError: to_locale('GerMAN') = 'gerMAN' has inconsistent casing
Falsifying explicit example: test_to_locale_case_handling_without_dash(
    language='GerMAN',
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction case for django.utils.translation.to_locale bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.utils.translation import to_locale

# Test cases showing the bug
test_cases = [
    'GerMAN',
    'EnGlish',
    'FrEnch',
    'UPPERCASE',
    'MiXeDcAsE',
    'lower',
    'en',
    'EN',
    'fr-FR',
    'en-US',
    'EN-US',
    'de-DE-1996'
]

print("Testing to_locale() function:")
print("=" * 50)

for test in test_cases:
    result = to_locale(test)
    print(f"to_locale('{test:15}') = '{result}'")

print("\n" + "=" * 50)
print("\nExpected behavior for non-hyphenated inputs:")
print("The entire string should be lowercase since line 233")
print("computes: lang = language.lower().partition('-')[0]")
print("But line 235 incorrectly uses: language[:3].lower() + language[3:]")
print("\nThis causes inconsistent casing like 'gerMAN' from 'GerMAN'")
```

<details>

<summary>
Inconsistent case handling demonstrated
</summary>
```
Testing to_locale() function:
==================================================
to_locale('GerMAN         ') = 'gerMAN'
to_locale('EnGlish        ') = 'english'
to_locale('FrEnch         ') = 'french'
to_locale('UPPERCASE      ') = 'uppERCASE'
to_locale('MiXeDcAsE      ') = 'mixeDcAsE'
to_locale('lower          ') = 'lower'
to_locale('en             ') = 'en'
to_locale('EN             ') = 'en'
to_locale('fr-FR          ') = 'fr_FR'
to_locale('en-US          ') = 'en_US'
to_locale('EN-US          ') = 'en_US'
to_locale('de-DE-1996     ') = 'de_DE-1996'

==================================================

Expected behavior for non-hyphenated inputs:
The entire string should be lowercase since line 233
computes: lang = language.lower().partition('-')[0]
But line 235 incorrectly uses: language[:3].lower() + language[3:]

This causes inconsistent casing like 'gerMAN' from 'GerMAN'
```
</details>

## Why This Is A Bug

The function contains a clear programming error where a computed variable is never used. On line 233 of `/django/utils/translation/__init__.py`, the function computes `lang = language.lower().partition("-")[0]`, which correctly lowercases the entire language code. However, when no country code is present (no hyphen), line 235 incorrectly returns `language[:3].lower() + language[3:]` instead of using the already-computed `lang` variable.

This creates inconsistent behavior:
- For inputs with hyphens (e.g., 'EN-US'), the function correctly normalizes to 'en_US'
- For inputs without hyphens longer than 3 characters (e.g., 'GerMAN'), only the first 3 characters are lowercased, resulting in 'gerMAN'
- Standard 2-3 character language codes work correctly by coincidence

The unused `lang` variable is clear evidence this is unintentional. The function attempts to normalize case but does so incorrectly for non-hyphenated inputs longer than 3 characters.

## Relevant Context

- **ISO 639 Language Codes**: Standard language codes are 2-3 lowercase letters (e.g., 'en', 'fr', 'ger'), so this bug rarely affects real-world usage
- **RFC 5646**: Specifies that language tags should be treated as case-insensitive, supporting the need for consistent normalization
- **Django Usage**: This function is used internally in Django's internationalization system to convert language codes to locale identifiers
- **Code Location**: `/django/utils/translation/__init__.py`, lines 231-244
- **Django Documentation**: https://docs.djangoproject.com/en/stable/topics/i18n/

The bug exists in all recent Django versions and has likely been present for years without causing issues due to the rarity of non-standard language codes in production use.

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