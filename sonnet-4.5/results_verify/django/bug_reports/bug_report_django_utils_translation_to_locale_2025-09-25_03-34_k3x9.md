# Bug Report: django.utils.translation.to_locale Incomplete Case Normalization

**Target**: `django.utils.translation.to_locale`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `to_locale()` function incorrectly handles uppercase characters beyond the third position when no dash is present in the input, resulting in inconsistent case normalization.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.utils.translation import to_locale


@given(st.text(min_size=4, max_size=10, alphabet=st.characters(min_codepoint=65, max_codepoint=90)))
def test_to_locale_without_dash_should_be_lowercase(language_str):
    result = to_locale(language_str)
    assert result == result.lower(), f"to_locale({language_str!r}) = {result!r}, but should be all lowercase when no dash present"
```

**Failing input**: `'AAAA'` (or any uppercase string with more than 3 characters and no dash)

## Reproducing the Bug

```python
from django.utils.translation import to_locale

print(to_locale('ENUS'))
print(to_locale('FRCA'))
print(to_locale('DEDE'))
```

Output:
```
enuS
frcA
dedE
```

Expected:
```
enus
frca
dede
```

## Why This Is A Bug

The function's docstring states it converts "a language name (en-us) into a locale name (en_US)". While the primary use case involves dashed language codes, the function should handle all inputs consistently.

When a language code without a dash is provided, the function attempts to normalize it to lowercase but fails to do so completely. This is because line 235 in `/django/utils/translation/__init__.py` uses:

```python
return language[:3].lower() + language[3:]
```

This only lowercases the first 3 characters, leaving characters after position 3 in their original case. Since line 233 already computes `lang, _, country = language.lower().partition("-")`, the variable `lang` contains the properly lowercased value but is not returned.

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