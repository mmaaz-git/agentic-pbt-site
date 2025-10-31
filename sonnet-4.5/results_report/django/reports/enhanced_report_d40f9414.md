# Bug Report: django.conf.locale Azerbaijani Incorrectly Marked as Right-to-Left

**Target**: `django.conf.locale.LANG_INFO['az']`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Azerbaijani language configuration in Django incorrectly sets `bidi: True`, marking it as a right-to-left language when it should be `bidi: False` since modern Azerbaijani uses Latin script which is left-to-right.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.conf.locale import LANG_INFO


def test_azerbaijani_bidi_bug():
    az_info = LANG_INFO['az']

    assert az_info['bidi'] == False, (
        f"Azerbaijani (az) is incorrectly marked as bidi=True. "
        f"Azerbaijani has used Latin script since 1991 and is a left-to-right language."
    )

if __name__ == "__main__":
    test_azerbaijani_bidi_bug()
    print("Test passed!")
```

<details>

<summary>
**Failing input**: `LANG_INFO['az']` returns `{'bidi': True, 'code': 'az', 'name': 'Azerbaijani', 'name_local': 'Azərbaycanca'}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 16, in <module>
    test_azerbaijani_bidi_bug()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 10, in test_azerbaijani_bidi_bug
    assert az_info['bidi'] == False, (
           ^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Azerbaijani (az) is incorrectly marked as bidi=True. Azerbaijani has used Latin script since 1991 and is a left-to-right language.
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.conf.locale import LANG_INFO

# Get Azerbaijani locale info
az_info = LANG_INFO['az']

# Display the current configuration
print("Azerbaijani locale configuration:")
print(f"  Code: {az_info['code']}")
print(f"  Name: {az_info['name']}")
print(f"  Name Local: {az_info['name_local']}")
print(f"  Bidi (Right-to-Left): {az_info['bidi']}")

print("\nComparing with other RTL languages:")
for lang_code in ['ar', 'fa', 'he', 'ur']:
    lang_info = LANG_INFO[lang_code]
    print(f"  {lang_code}: name_local='{lang_info['name_local']}' (bidi={lang_info['bidi']})")

print("\nThe Issue:")
print("Azerbaijani is marked as bidi=True (right-to-left) but uses Latin script.")
print("The name_local 'Azərbaycanca' is written in Latin characters.")
print("All other RTL languages use actual RTL scripts (Arabic, Hebrew, etc.).")

# Demonstrate the error
assert az_info['bidi'] == True, "Expected bidi=True (current state)"
print("\nCurrent assertion passes: az_info['bidi'] == True")

# What it should be
print("\nWhat it should be: az_info['bidi'] == False")
print("Because Azerbaijani uses Latin script which is left-to-right.")
```

<details>

<summary>
Azerbaijani incorrectly marked as RTL despite using Latin script
</summary>
```
Azerbaijani locale configuration:
  Code: az
  Name: Azerbaijani
  Name Local: Azərbaycanca
  Bidi (Right-to-Left): True

Comparing with other RTL languages:
  ar: name_local='العربيّة' (bidi=True)
  fa: name_local='فارسی' (bidi=True)
  he: name_local='עברית' (bidi=True)
  ur: name_local='اردو' (bidi=True)

The Issue:
Azerbaijani is marked as bidi=True (right-to-left) but uses Latin script.
The name_local 'Azərbaycanca' is written in Latin characters.
All other RTL languages use actual RTL scripts (Arabic, Hebrew, etc.).

Current assertion passes: az_info['bidi'] == True

What it should be: az_info['bidi'] == False
Because Azerbaijani uses Latin script which is left-to-right.
```
</details>

## Why This Is A Bug

This is a clear data error in Django's language configuration. The evidence is overwhelming:

1. **Azerbaijani is the ONLY language marked as `bidi=True` that displays its `name_local` in Latin script**. All other RTL languages in Django (Arabic, Persian, Hebrew, Urdu, Uyghur, Central Kurdish) show their `name_local` in actual right-to-left scripts.

2. **The `name_local` field "Azərbaycanca" uses Latin characters** with specific diacritics (ə, ç) that are part of the modern Azerbaijani Latin alphabet officially adopted by the Republic of Azerbaijan on December 25, 1991.

3. **Latin script is always left-to-right (LTR)**, never right-to-left. This is a fundamental characteristic of the Latin writing system.

4. **This causes actual rendering problems** in Django applications serving Azerbaijani content:
   - Text will be incorrectly aligned to the right instead of left
   - Form inputs will have reversed text direction
   - UI layouts may be mirrored inappropriately
   - Mixed LTR/RTL content will be handled incorrectly

5. **Historical context**: While Azerbaijani historically used Arabic script (which is RTL) until the 1920s, and some regions like Iran still use it, the Django locale clearly represents the modern Republic of Azerbaijan variant based on the Latin script spelling in `name_local`.

## Relevant Context

Django's `bidi` field in `LANG_INFO` controls text directionality throughout the framework. When `bidi=True`, Django:
- Sets `dir="rtl"` attributes in HTML templates
- Applies RTL-specific CSS rules
- Mirrors UI components for RTL layout
- Adjusts text alignment in forms and widgets

The current languages correctly marked as RTL in Django (`/django/conf/locale/__init__.py:35-585`):
- Arabic (ar): العربيّة
- Algerian Arabic (ar-dz): العربية الجزائرية
- Central Kurdish (ckb): کوردی
- Persian (fa): فارسی
- Hebrew (he): עברית
- Uyghur (ug): ئۇيغۇرچە
- Urdu (ur): اردو

Note how each shows its native name in an actual RTL script, unlike Azerbaijani's "Azərbaycanca" in Latin.

## Proposed Fix

```diff
--- a/django/conf/locale/__init__.py
+++ b/django/conf/locale/__init__.py
@@ -34,7 +34,7 @@ LANG_INFO = {
         "name_local": "asturianu",
     },
     "az": {
-        "bidi": True,
+        "bidi": False,
         "code": "az",
         "name": "Azerbaijani",
         "name_local": "Azərbaycanca",
```