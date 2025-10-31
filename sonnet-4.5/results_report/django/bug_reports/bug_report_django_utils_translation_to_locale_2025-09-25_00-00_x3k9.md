# Bug Report: django.utils.translation.to_locale Case Handling

**Target**: `django.utils.translation.to_locale`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `to_locale()` function has inconsistent case handling when processing language codes without a country part. It lowercases only the first 3 characters while leaving the rest unchanged, resulting in mixed-case output like `'gerMAN'` for input `'GerMAN'`.

## Property-Based Test

```python
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
```

**Failing input**: `'GerMAN'` (and other mixed-case strings without `-`)

## Reproducing the Bug

```python
from django.utils.translation import to_locale

print(to_locale('GerMAN'))
print(to_locale('EnGlish'))
print(to_locale('FrEnch'))
```

## Why This Is A Bug

The function has inconsistent behavior:
- Line 233 correctly lowercases the input: `lang, _, country = language.lower().partition("-")`
- But line 235 uses the original `language` variable instead of the lowercased version: `return language[:3].lower() + language[3:]`

This creates mixed-case results like `'gerMAN'` from `'GerMAN'`, where the first 3 characters are lowercase but the rest retain their original casing.

While real-world language codes are typically lowercase, the function should handle any input consistently since it already attempts to normalize casing.

## Fix

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