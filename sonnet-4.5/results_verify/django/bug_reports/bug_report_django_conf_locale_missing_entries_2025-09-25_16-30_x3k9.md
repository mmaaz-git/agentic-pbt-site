# Bug Report: django.conf.locale Missing LANG_INFO Entries for 7 Locales

**Target**: `django.conf.locale.LANG_INFO`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `LANG_INFO` dictionary in `django.conf.locale` is missing entries for 7 locale variants that have corresponding directories with locale data on disk: `de-ch`, `en-ca`, `en-ie`, `es-pr`, `fr-be`, `fr-ca`, and `fr-ch`.

## Property-Based Test

```python
import os
from django.conf import locale
from hypothesis import given, strategies as st, settings

LANG_INFO = locale.LANG_INFO
locale_dir = os.path.dirname(locale.__file__)

locale_dirs_on_disk = []
for item in os.listdir(locale_dir):
    item_path = os.path.join(locale_dir, item)
    if os.path.isdir(item_path) and not item.startswith('_'):
        locale_dirs_on_disk.append(item)


def normalize_locale_code(dir_name):
    return dir_name.lower().replace('_', '-')


@given(st.sampled_from(sorted(locale_dirs_on_disk)))
@settings(max_examples=len(locale_dirs_on_disk))
def test_all_locale_directories_have_lang_info(locale_dir_name):
    normalized = normalize_locale_code(locale_dir_name)
    assert normalized in LANG_INFO
```

**Failing input**: `de_CH` (Swiss German), along with 6 other locales

## Reproducing the Bug

```python
import os
from django.conf import locale

LANG_INFO = locale.LANG_INFO
locale_dir = os.path.dirname(locale.__file__)

locale_dirs = [item for item in os.listdir(locale_dir)
               if os.path.isdir(os.path.join(locale_dir, item)) and not item.startswith('_')]

def normalize_code(dir_name):
    return dir_name.lower().replace('_', '-')

missing = []
for dir_name in sorted(locale_dirs):
    normalized = normalize_code(dir_name)
    if normalized not in LANG_INFO:
        locale_path = os.path.join(locale_dir, dir_name)
        has_formats = os.path.exists(os.path.join(locale_path, 'formats.py'))
        missing.append((dir_name, normalized, has_formats))

print(f"Found {len(missing)} locale directories without LANG_INFO entries:\n")
for dir_name, normalized, has_formats in missing:
    print(f"  {dir_name:10s} (normalized: {normalized:10s}) - has formats: {has_formats}")
```

Output:
```
Found 7 locale directories without LANG_INFO entries:

  de_CH      (normalized: de-ch     ) - has formats: True
  en_CA      (normalized: en-ca     ) - has formats: True
  en_IE      (normalized: en-ie     ) - has formats: True
  es_PR      (normalized: es-pr     ) - has formats: True
  fr_BE      (normalized: fr-be     ) - has formats: True
  fr_CA      (normalized: fr-ca     ) - has formats: True
  fr_CH      (normalized: fr-ch     ) - has formats: True
```

## Why This Is A Bug

The module docstring for `django.conf.locale` states that "LANG_INFO is a dictionary structure to provide meta information about languages." However, it fails to provide metadata for 7 locale variants that have full locale directories with format definitions:

1. `de-ch` (Swiss German)
2. `en-ca` (Canadian English)
3. `en-ie` (Irish English)
4. `es-pr` (Puerto Rican Spanish)
5. `fr-be` (Belgian French)
6. `fr-ca` (Canadian French)
7. `fr-ch` (Swiss French)

Each of these directories contains a `formats.py` file with locale-specific formatting rules, indicating they are legitimate, supported locales. Users who need metadata about these locales (such as bidirectional text settings, English names, or local names) cannot retrieve it from `LANG_INFO`, even though the locales are fully supported by Django.

This creates an inconsistency where Django has locale data for these variants but doesn't expose their metadata through the canonical `LANG_INFO` dictionary.

## Fix

Add the missing entries to `LANG_INFO` in `django/conf/locale/__init__.py`. Based on the pattern of existing entries, the fix would add:

```diff
--- a/django/conf/locale/__init__.py
+++ b/django/conf/locale/__init__.py
@@ -104,6 +104,11 @@ LANG_INFO = {
         "name": "German",
         "name_local": "Deutsch",
     },
+    "de-ch": {
+        "bidi": False,
+        "code": "de-ch",
+        "name": "Swiss High German",
+        "name_local": "Schweizer Hochdeutsch",
+    },
     "dsb": {
         "bidi": False,
@@ -130,6 +135,16 @@ LANG_INFO = {
         "name": "British English",
         "name_local": "British English",
     },
+    "en-ca": {
+        "bidi": False,
+        "code": "en-ca",
+        "name": "Canadian English",
+        "name_local": "Canadian English",
+    },
+    "en-ie": {
+        "bidi": False,
+        "code": "en-ie",
+        "name": "Irish English",
+        "name_local": "Irish English",
+    },
     "eo": {
         "bidi": False,
@@ -176,6 +191,11 @@ LANG_INFO = {
         "name": "Venezuelan Spanish",
         "name_local": "español de Venezuela",
     },
+    "es-pr": {
+        "bidi": False,
+        "code": "es-pr",
+        "name": "Puerto Rican Spanish",
+        "name_local": "español de Puerto Rico",
+    },
     "et": {
         "bidi": False,
@@ -206,6 +226,21 @@ LANG_INFO = {
         "name": "French",
         "name_local": "français",
     },
+    "fr-be": {
+        "bidi": False,
+        "code": "fr-be",
+        "name": "Belgian French",
+        "name_local": "français de Belgique",
+    },
+    "fr-ca": {
+        "bidi": False,
+        "code": "fr-ca",
+        "name": "Canadian French",
+        "name_local": "français canadien",
+    },
+    "fr-ch": {
+        "bidi": False,
+        "code": "fr-ch",
+        "name": "Swiss French",
+        "name_local": "français de Suisse",
+    },
     "fy": {
         "bidi": False,
```