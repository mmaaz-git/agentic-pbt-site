# Bug Report: django.utils.translation.to_locale Case Sensitivity Without Dash

**Target**: `django.utils.translation.to_locale`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `to_locale` function incorrectly preserves case from the fourth character onwards for language codes without dashes, while correctly normalizing case for codes with dashes, violating the expected case-insensitive behavior.

## Property-Based Test

```python
from hypothesis import assume, given, strategies as st
from django.utils.translation import to_locale

@given(st.text(alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=4, max_size=10))
def test_to_locale_case_invariance_no_dash(language):
    assume('-' not in language)

    result_upper = to_locale(language.upper())
    result_lower = to_locale(language.lower())

    assert result_upper == result_lower, f"to_locale should be case-insensitive"

if __name__ == "__main__":
    test_to_locale_case_invariance_no_dash()
```

<details>

<summary>
**Failing input**: `'AAAA'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 14, in <module>
    test_to_locale_case_invariance_no_dash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 5, in test_to_locale_case_invariance_no_dash
    def test_to_locale_case_invariance_no_dash(language):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 11, in test_to_locale_case_invariance_no_dash
    assert result_upper == result_lower, f"to_locale should be case-insensitive"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: to_locale should be case-insensitive
Falsifying example: test_to_locale_case_invariance_no_dash(
    language='AAAA',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from django.utils.translation import to_locale

# Test case demonstrating the bug with uppercase input without dash
print(f"to_locale('AAAA') = {to_locale('AAAA')!r}")
print(f"to_locale('aaaa') = {to_locale('aaaa')!r}")

# Show that they produce different results
assert to_locale('AAAA') == to_locale('aaaa'), "Case should not matter"
```

<details>

<summary>
AssertionError: Case should not matter
</summary>
```
to_locale('AAAA') = 'aaaA'
to_locale('aaaa') = 'aaaa'
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/31/repo.py", line 8, in <module>
    assert to_locale('AAAA') == to_locale('aaaa'), "Case should not matter"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Case should not matter
```
</details>

## Why This Is A Bug

The `to_locale` function is intended to normalize language codes to locale format, and it already demonstrates case-insensitive behavior for inputs containing dashes. For example, both `to_locale('EN-US')` and `to_locale('en-us')` correctly return `'en_US'`. However, when the input lacks a dash separator, the function fails to maintain this case-insensitive behavior.

The bug occurs in line 235 of `/django/utils/translation/__init__.py`. The function correctly lowercases the input on line 233 (`language.lower().partition("-")`), but then incorrectly uses the original `language` variable on line 235 instead of the lowercased result:

```python
lang, _, country = language.lower().partition("-")
if not country:
    return language[:3].lower() + language[3:]  # Bug: uses 'language' instead of lowercased version
```

This inconsistency violates the principle of uniform case handling. The inverse function `to_language` is fully case-insensitive (both `to_language('EN')` and `to_language('en')` return `'en'`), suggesting that `to_locale` should exhibit the same behavior.

## Relevant Context

The Django translation utilities are widely used in internationalization (i18n) workflows where language codes may come from various sources with inconsistent casing (HTTP headers, user preferences, configuration files, etc.). The inconsistent case handling can lead to subtle bugs in applications that expect normalized output regardless of input casing.

The function already handles several edge cases properly:
- Language codes with dashes and country codes (e.g., "en-US" → "en_US")
- Language codes with script subtags (e.g., "sr-latn" → "sr_Latn")
- Multiple subtags (e.g., "zh-hans-CN" → "zh_Hans-CN")

Documentation: The function's docstring states it turns "a language name (en-us) into a locale name (en_US)" but doesn't explicitly mention case sensitivity requirements.

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