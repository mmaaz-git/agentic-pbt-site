# Bug Report: django.utils.translation.to_locale Case Sensitivity

**Target**: `django.utils.translation.to_locale`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `to_locale` function produces inconsistent case-handling behavior when the input language name does not contain a dash separator, violating the principle of case-insensitive input handling that works correctly for dash-containing inputs.

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
```

**Failing input**: `'AAAA'`

## Reproducing the Bug

```python
from django.utils.translation import to_locale

print(f"to_locale('AAAA') = {to_locale('AAAA')!r}")
print(f"to_locale('aaaa') = {to_locale('aaaa')!r}")

assert to_locale('AAAA') == to_locale('aaaa'), "Case should not matter"
```

Output:
```
to_locale('AAAA') = 'aaaA'
to_locale('aaaa') = 'aaaa'
AssertionError: Case should not matter
```

## Why This Is A Bug

The function correctly handles case for inputs with dashes (e.g., `to_locale('EN-US')` and `to_locale('en-us')` both return `'en_US'`), but fails to do so for inputs without dashes. This inconsistency violates the expected behavior that the function should normalize case regardless of input format.

The root cause is in this code path:

```python
lang, _, country = language.lower().partition("-")
if not country:
    return language[:3].lower() + language[3:]
```

The bug is using the original `language` variable (which retains its original case) instead of the already-lowercased result from the partition operation.

## Fix

```diff
--- a/django/utils/translation/__init__.py
+++ b/django/utils/translation/__init__.py
@@ -247,7 +247,7 @@ def to_locale(language):
     """Turn a language name (en-us) into a locale name (en_US)."""
     lang, _, country = language.lower().partition("-")
     if not country:
-        return language[:3].lower() + language[3:]
+        return language.lower()
     # A language with > 2 characters after the dash only has its first
     # character after the dash capitalized; e.g. sr-latn becomes sr_Latn.
     # A language with 2 characters after the dash has both characters
```