# Bug Report: django.conf.locale Azerbaijani Incorrectly Marked as Right-to-Left

**Target**: `django.conf.locale.LANG_INFO['az']`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Azerbaijani language (code 'az') is incorrectly marked as a right-to-left (bidirectional) language with `bidi: True` in LANG_INFO, when it should be `bidi: False` since Azerbaijani has used Latin script exclusively since 1991.

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
```

**Failing input**: `LANG_INFO['az']` returns `{'bidi': True, 'code': 'az', 'name': 'Azerbaijani', 'name_local': 'Azərbaycanca'}`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.conf.locale import LANG_INFO

az_info = LANG_INFO['az']
print(f"Azerbaijani bidi value: {az_info['bidi']}")

assert az_info['bidi'] == True
```

## Why This Is A Bug

Azerbaijani (Azərbaycan dili) is a Turkic language that historically used Arabic script but transitioned to Latin script in 1991 in Azerbaijan (and uses Cyrillic in some regions). The Latin script version, which is the standard in Azerbaijan today, is left-to-right (LTR), not right-to-left (RTL).

The `bidi` field should be `False` for Latin-script languages. Having it marked as `True` will cause text rendering issues in Django applications that serve Azerbaijani content, as browsers and text rendering engines will incorrectly apply right-to-left text direction.

This affects any Django application using the Azerbaijani locale, potentially causing:
- Incorrect text alignment in templates
- Wrong text direction in forms and UI elements
- Improper handling of mixed LTR/RTL content

## Fix

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