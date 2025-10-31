# Bug Report: django.conf.locale Missing Base Chinese Language Code

**Target**: `django.conf.locale.LANG_INFO`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

LANG_INFO contains Chinese language variants (`zh-hans`, `zh-hant`) but is missing the base Chinese language code `zh`. This breaks Django's language fallback logic and prevents users from using the generic `zh` language code.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.conf.locale import LANG_INFO

@given(st.sampled_from(list(LANG_INFO.keys())))
def test_language_variants_have_base_language(lang_code):
    if '-' in lang_code:
        base_lang = lang_code.split('-')[0]
        info = LANG_INFO[lang_code]

        if 'fallback' in info and not ('name' in info):
            return

        assert base_lang in LANG_INFO, \
            f"Language variant {lang_code} exists but base language {base_lang} is not in LANG_INFO"
```

**Failing input**: `lang_code='zh-hans'`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.conf import settings
settings.configure(USE_I18N=True)

from django.conf.locale import LANG_INFO
from django.utils.translation import get_language_info

print("1. zh-hans exists:", 'zh-hans' in LANG_INFO)
print("2. zh-hant exists:", 'zh-hant' in LANG_INFO)
print("3. zh exists:", 'zh' in LANG_INFO)

try:
    get_language_info('zh')
except KeyError as e:
    print(f"4. get_language_info('zh') fails: {e}")

try:
    get_language_info('zh-unknown')
except KeyError as e:
    print(f"5. get_language_info('zh-unknown') fails: {e}")
```

Output:
```
1. zh-hans exists: True
2. zh-hant exists: True
3. zh exists: False
4. get_language_info('zh') fails: Unknown language code zh.
5. get_language_info('zh-unknown') fails: Unknown language code zh-unknown and zh.
```

## Why This Is A Bug

1. **Breaks fallback logic**: Django's `get_language_info()` function has fallback logic that splits language codes like `zh-unknown` on the hyphen and tries to look up the base language `zh`. This fails because `zh` is not in LANG_INFO.

2. **Standard language code unsupported**: `zh` is a valid and commonly used ISO 639-1 language code for Chinese. Users cannot use this standard code.

3. **Inconsistent with other languages**: All other language variants in LANG_INFO (like `en-gb`, `es-mx`, `fr-ca`) have their base language defined (`en`, `es`, `fr`). Only Chinese variants lack their base language.

4. **Confusing error messages**: When users request an unknown Chinese variant, they get an error about both the variant AND the base language being unknown, which is confusing.

## Fix

Add a base Chinese language entry to LANG_INFO. Since Chinese has two main writing systems (Simplified and Traditional) without a clear "default", the base `zh` entry should use a fallback to one of them, similar to how `zh-cn` already falls back to `zh-hans`:

```diff
--- a/django/conf/locale/__init__.py
+++ b/django/conf/locale/__init__.py
@@ -596,6 +596,12 @@ LANG_INFO = {
         "name": "Vietnamese",
         "name_local": "Tiếng Việt",
     },
+    "zh": {
+        "bidi": False,
+        "code": "zh",
+        "name": "Chinese",
+        "name_local": "中文",
+    },
     "zh-cn": {
         "fallback": ["zh-hans"],
     },
```

Alternatively, if Django prefers to use fallback-only entries for ambiguous base languages, use:

```diff
+    "zh": {
+        "fallback": ["zh-hans"],
+    },
```

This follows the same pattern as `zh-cn` and makes `zh` a valid language code that falls back to Simplified Chinese.