# Bug Report: django.conf.locale Azerbaijani Incorrectly Configured as Right-to-Left Language

**Target**: `django.conf.locale.LANG_INFO`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Azerbaijani language ('az') is incorrectly configured with `bidi=True` in Django's LANG_INFO dictionary, causing it to render as right-to-left text despite using Latin script which is strictly left-to-right.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test using Hypothesis that reveals the Django Azerbaijani bidi bug.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

import django.conf.locale as locale_module
from hypothesis import given, strategies as st


def is_rtl_script(text):
    """Check if text contains RTL script characters."""
    rtl_ranges = [
        (0x0590, 0x05FF),  # Hebrew
        (0x0600, 0x06FF),  # Arabic
        (0x0700, 0x074F),  # Syriac
        (0x0750, 0x077F),  # Arabic Supplement
        (0x0780, 0x07BF),  # Thaana
        (0x07C0, 0x07FF),  # N'Ko
        (0x0800, 0x083F),  # Samaritan
        (0x0840, 0x085F),  # Mandaic
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB1D, 0xFB4F),  # Hebrew presentation forms
        (0xFB50, 0xFDFF),  # Arabic presentation forms A
        (0xFE70, 0xFEFF),  # Arabic presentation forms B
    ]

    return any(
        any(start <= ord(c) <= end for start, end in rtl_ranges)
        for c in text
    )


@given(st.sampled_from(list(locale_module.LANG_INFO.keys())))
def test_bidi_languages_use_rtl_script(lang_code):
    """Test that languages marked as bidi use RTL scripts."""
    info = locale_module.LANG_INFO[lang_code]

    if info.get('bidi', False):
        name_local = info.get('name_local', '')

        assert is_rtl_script(name_local), \
            f"Language {lang_code} marked as bidi but name_local '{name_local}' " \
            f"doesn't use RTL script"


if __name__ == '__main__':
    # Run the test
    test_bidi_languages_use_rtl_script()
```

<details>

<summary>
**Failing input**: `lang_code='az'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 51, in <module>
    test_bidi_languages_use_rtl_script()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 37, in test_bidi_languages_use_rtl_script
    def test_bidi_languages_use_rtl_script(lang_code):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/45/hypo.py", line 44, in test_bidi_languages_use_rtl_script
    assert is_rtl_script(name_local), \
           ~~~~~~~~~~~~~^^^^^^^^^^^^
AssertionError: Language az marked as bidi but name_local 'Azərbaycanca' doesn't use RTL script
Falsifying example: test_bidi_languages_use_rtl_script(
    lang_code='az',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/45/hypo.py:45
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction script for Django Azerbaijani bidi bug.
This demonstrates that Azerbaijani (az) is incorrectly marked as bidi=True
despite using Latin script which is left-to-right.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.conf.locale import LANG_INFO

# Get Azerbaijani language info
az_info = LANG_INFO['az']

print("=== Azerbaijani Language Configuration ===")
print(f"Language code: az")
print(f"Name: {az_info['name']}")
print(f"Name (local): {az_info['name_local']}")
print(f"Marked as bidi (RTL): {az_info['bidi']}")
print()

# Show the characters in the local name
print("=== Character Analysis of 'Azərbaycanca' ===")
for i, char in enumerate(az_info['name_local']):
    print(f"  Position {i}: '{char}' (U+{ord(char):04X}) - {char.isalpha() and 'Letter' or 'Other'}")
print()

# Compare with other bidi languages
print("=== Comparison with Other Bidi Languages ===")
bidi_langs = [(code, info) for code, info in LANG_INFO.items()
              if isinstance(info, dict) and info.get('bidi', False)]

for code, info in sorted(bidi_langs):
    name_local = info.get('name_local', '')
    print(f"{code:6} | bidi={info['bidi']} | name_local='{name_local}'")
print()

# Check if Azerbaijani contains any RTL script characters
def contains_rtl_characters(text):
    """Check if text contains any RTL script characters."""
    rtl_ranges = [
        (0x0590, 0x05FF),  # Hebrew
        (0x0600, 0x06FF),  # Arabic
        (0x0700, 0x074F),  # Syriac
        (0x0750, 0x077F),  # Arabic Supplement
        (0x0780, 0x07BF),  # Thaana
        (0x07C0, 0x07FF),  # N'Ko
        (0x0800, 0x083F),  # Samaritan
        (0x0840, 0x085F),  # Mandaic
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB1D, 0xFB4F),  # Hebrew presentation forms
        (0xFB50, 0xFDFF),  # Arabic presentation forms A
        (0xFE70, 0xFEFF),  # Arabic presentation forms B
    ]

    for char in text:
        code_point = ord(char)
        for start, end in rtl_ranges:
            if start <= code_point <= end:
                return True
    return False

print("=== RTL Script Analysis ===")
for code, info in sorted(bidi_langs):
    name_local = info.get('name_local', '')
    has_rtl = contains_rtl_characters(name_local)
    status = "✓ CORRECT" if has_rtl else "✗ INCORRECT"
    print(f"{code:6} | name_local='{name_local}' | Contains RTL: {has_rtl} | {status}")

print("\n=== CONCLUSION ===")
print(f"Azerbaijani ('az') is marked as bidi=True but its name_local '{az_info['name_local']}'")
print(f"contains NO RTL script characters. This is a BUG.")
print(f"The language should be marked as bidi=False since it uses Latin script.")
```

<details>

<summary>
Azerbaijani incorrectly marked as bidi despite using Latin script
</summary>
```
=== Azerbaijani Language Configuration ===
Language code: az
Name: Azerbaijani
Name (local): Azərbaycanca
Marked as bidi (RTL): True

=== Character Analysis of 'Azərbaycanca' ===
  Position 0: 'A' (U+0041) - Letter
  Position 1: 'z' (U+007A) - Letter
  Position 2: 'ə' (U+0259) - Letter
  Position 3: 'r' (U+0072) - Letter
  Position 4: 'b' (U+0062) - Letter
  Position 5: 'a' (U+0061) - Letter
  Position 6: 'y' (U+0079) - Letter
  Position 7: 'c' (U+0063) - Letter
  Position 8: 'a' (U+0061) - Letter
  Position 9: 'n' (U+006E) - Letter
  Position 10: 'c' (U+0063) - Letter
  Position 11: 'a' (U+0061) - Letter

=== Comparison with Other Bidi Languages ===
ar     | bidi=True | name_local='العربيّة'
ar-dz  | bidi=True | name_local='العربية الجزائرية'
az     | bidi=True | name_local='Azərbaycanca'
ckb    | bidi=True | name_local='کوردی'
fa     | bidi=True | name_local='فارسی'
he     | bidi=True | name_local='עברית'
ug     | bidi=True | name_local='ئۇيغۇرچە'
ur     | bidi=True | name_local='اردو'

=== RTL Script Analysis ===
ar     | name_local='العربيّة' | Contains RTL: True | ✓ CORRECT
ar-dz  | name_local='العربية الجزائرية' | Contains RTL: True | ✓ CORRECT
az     | name_local='Azərbaycanca' | Contains RTL: False | ✗ INCORRECT
ckb    | name_local='کوردی' | Contains RTL: True | ✓ CORRECT
fa     | name_local='فارسی' | Contains RTL: True | ✓ CORRECT
he     | name_local='עברית' | Contains RTL: True | ✓ CORRECT
ug     | name_local='ئۇيغۇرچە' | Contains RTL: True | ✓ CORRECT
ur     | name_local='اردو' | Contains RTL: True | ✓ CORRECT

=== CONCLUSION ===
Azerbaijani ('az') is marked as bidi=True but its name_local 'Azərbaycanca'
contains NO RTL script characters. This is a BUG.
The language should be marked as bidi=False since it uses Latin script.
```
</details>

## Why This Is A Bug

This violates Django's expected behavior in several critical ways:

1. **Inconsistent Configuration Pattern**: Out of 8 languages marked with `bidi=True` in LANG_INFO, Azerbaijani is the ONLY one whose `name_local` field uses left-to-right script. All 7 other languages (Arabic, Algerian Arabic, Central Kurdish, Persian, Hebrew, Uyghur, and Urdu) correctly use right-to-left scripts (Arabic or Hebrew).

2. **Character Analysis Confirms Latin Script**: The name "Azərbaycanca" consists entirely of Latin characters:
   - Standard Latin letters: A, z, r, b, a, y, c, n
   - Latin Extended character: ə (U+0259, Latin Small Letter Schwa)
   - No characters from RTL Unicode ranges (Hebrew, Arabic, etc.)

3. **Incorrect UI Rendering**: When Django templates use `{% get_current_language_bidi %}` for Azerbaijani, they incorrectly apply `dir="rtl"` and right-aligned CSS styles, causing:
   - Text to be right-aligned instead of left-aligned
   - Potential reading order issues in complex layouts
   - Broken user interface for Azerbaijani users

4. **Historical Context Doesn't Justify Current Setting**: While Azerbaijani historically used Arabic script (until 1929), it has used Latin script officially since 1991 - over 30 years. Django's own data acknowledges this by using the modern Latin spelling "Azərbaycanca" in the `name_local` field.

## Relevant Context

The `bidi` flag in Django's LANG_INFO is used throughout the framework's internationalization system to determine text directionality. Key usage points include:

- Template tag `{% get_current_language_bidi %}` returns this value
- Admin interface uses it to set HTML `dir` attribute
- Forms and widgets adjust their rendering based on this flag

Django's source code location: `/django/conf/locale/__init__.py` (lines 35-40 in the current version)

The pattern is clear from examining all bidi languages in LANG_INFO:
- Languages with `bidi=True` should use scripts that are inherently right-to-left
- The `name_local` field serves as the canonical representation of how the language writes its own name
- All other bidi languages follow this pattern correctly

## Proposed Fix

```diff
--- a/django/conf/locale/__init__.py
+++ b/django/conf/locale/__init__.py
@@ -33,7 +33,7 @@ LANG_INFO = {
         "name_local": "asturianu",
     },
     "az": {
-        "bidi": True,
+        "bidi": False,
         "code": "az",
         "name": "Azerbaijani",
         "name_local": "Azərbaycanca",
```