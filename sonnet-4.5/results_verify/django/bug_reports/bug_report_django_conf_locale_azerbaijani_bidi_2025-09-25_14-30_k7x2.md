# Bug Report: django.conf.locale Azerbaijani Incorrectly Marked as Bidi

**Target**: `django.conf.locale.LANG_INFO`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Azerbaijani language (code 'az') is incorrectly marked as bidirectional (`bidi: True`) in LANG_INFO, despite using Latin script which is left-to-right.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import django.conf.locale as locale_module
from hypothesis import given, strategies as st


def is_rtl_script(text):
    rtl_ranges = [
        (0x0590, 0x05FF),
        (0x0600, 0x06FF),
        (0x0700, 0x074F),
        (0x0750, 0x077F),
        (0x0780, 0x07BF),
        (0x07C0, 0x07FF),
        (0x0800, 0x083F),
        (0x0840, 0x085F),
        (0x08A0, 0x08FF),
        (0xFB1D, 0xFB4F),
        (0xFB50, 0xFDFF),
        (0xFE70, 0xFEFF),
    ]

    return any(
        any(start <= ord(c) <= end for start, end in rtl_ranges)
        for c in text
    )


@given(st.sampled_from(list(locale_module.LANG_INFO.keys())))
def test_bidi_languages_use_rtl_script(lang_code):
    info = locale_module.LANG_INFO[lang_code]

    if info.get('bidi', False):
        name_local = info.get('name_local', '')

        assert is_rtl_script(name_local), \
            f"Language {lang_code} marked as bidi but name_local '{name_local}' " \
            f"doesn't use RTL script"
```

**Failing input**: `lang_code='az'`

## Reproducing the Bug

```python
from django.conf.locale import LANG_INFO

az_info = LANG_INFO['az']

print(f"Azerbaijani language info: {az_info}")
print(f"Marked as bidi (RTL): {az_info['bidi']}")
print(f"Name (local): {az_info['name_local']}")

for code in ['ar', 'fa', 'he', 'ur']:
    info = LANG_INFO[code]
    print(f"{code}: bidi={info['bidi']}, name_local='{info['name_local']}'")
```

## Why This Is A Bug

1. Modern Azerbaijani uses the Latin script (since 1991), which is left-to-right
2. The `name_local` field contains "Azərbaycanca" - clearly Latin characters with diacritics
3. All other languages marked as `bidi: True` use right-to-left scripts (Arabic or Hebrew) as evidenced by their `name_local` values
4. This will cause Django applications to incorrectly render Azerbaijani text as right-to-left in the UI

Historical note: While Azerbaijani was written in Arabic script until 1929, it transitioned to Cyrillic (1939-1991) and then to Latin script (1991-present).

## Fix

```diff
--- a/django/conf/locale/__init__.py
+++ b/django/conf/locale/__init__.py
@@ -35,7 +35,7 @@ LANG_INFO = {
         "name_local": "asturianu",
     },
     "az": {
-        "bidi": True,
+        "bidi": False,
         "code": "az",
         "name": "Azerbaijani",
         "name_local": "Azərbaycanca",
```