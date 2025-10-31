# Bug Report: django.conf.locale Missing Base Chinese Language Code

**Target**: `django.conf.locale.LANG_INFO`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Django's LANG_INFO dictionary contains Chinese language variants (`zh-hans`, `zh-hant`) but lacks the base Chinese language code `zh`, breaking Django's language fallback mechanism and preventing users from using the standard ISO 639-1 Chinese language code.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.conf import settings
settings.configure(USE_I18N=True)

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

if __name__ == "__main__":
    test_language_variants_have_base_language()
```

<details>

<summary>
**Failing input**: `lang_code='zh-hans'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 23, in <module>
    test_language_variants_have_base_language()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 11, in test_language_variants_have_base_language
    def test_language_variants_have_base_language(lang_code):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 19, in test_language_variants_have_base_language
    assert base_lang in LANG_INFO, \
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Language variant zh-hans exists but base language zh is not in LANG_INFO
Falsifying example: test_language_variants_have_base_language(
    lang_code='zh-hans',
)
```
</details>

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
    info = get_language_info('zh')
    print(f"4. get_language_info('zh') returned: {info}")
except KeyError as e:
    print(f"4. get_language_info('zh') fails: {e}")

try:
    info = get_language_info('zh-unknown')
    print(f"5. get_language_info('zh-unknown') returned: {info}")
except KeyError as e:
    print(f"5. get_language_info('zh-unknown') fails: {e}")

print("\nComparing with other language variants that work correctly:")
print("6. en exists:", 'en' in LANG_INFO)
print("7. en-gb exists:", 'en-gb' in LANG_INFO)

try:
    info = get_language_info('en-unknown')
    print(f"8. get_language_info('en-unknown') returned: code={info.get('code')}, name={info.get('name')}")
except KeyError as e:
    print(f"8. get_language_info('en-unknown') fails: {e}")
```

<details>

<summary>
KeyError raised when requesting Chinese language codes
</summary>
```
1. zh-hans exists: True
2. zh-hant exists: True
3. zh exists: False
4. get_language_info('zh') fails: 'Unknown language code zh.'
5. get_language_info('zh-unknown') fails: 'Unknown language code zh-unknown and zh.'

Comparing with other language variants that work correctly:
6. en exists: True
7. en-gb exists: True
8. get_language_info('en-unknown') returned: code=en, name=English
```
</details>

## Why This Is A Bug

Django's `get_language_info()` function implements a fallback mechanism where it splits language codes containing hyphens and attempts to look up the base language when the full code isn't found. This is documented behavior in the Django translation system, as seen in `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/utils/translation/__init__.py:269-287`.

The bug violates expected behavior in several ways:

1. **Breaks documented fallback logic**: When requesting an unknown Chinese variant like `zh-unknown`, Django tries to fall back to the base language `zh`, but this fails because `zh` is missing from LANG_INFO. The error message "Unknown language code zh-unknown and zh" reveals this failed fallback attempt.

2. **Inconsistent with all other languages**: Every other language with variants in LANG_INFO has its base language defined. For example:
   - `en-gb`, `en-au` → base language `en` exists
   - `es-ar`, `es-co`, `es-mx` → base language `es` exists
   - `pt-br` → base language `pt` exists
   - `fr-ca` (through Django's locales) → base language `fr` exists

   Chinese is the only exception where variants (`zh-hans`, `zh-hant`) exist without their base language.

3. **Prevents use of standard language codes**: `zh` is the valid ISO 639-1 language code for Chinese, widely used in web applications. Users cannot use this standard code in Django applications.

4. **Causes confusing error messages**: When users try to use any unknown Chinese variant, they get an error mentioning both the variant AND the base language being unknown, which is misleading since the issue is specifically the missing `zh` entry.

## Relevant Context

The LANG_INFO dictionary is defined in `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/conf/locale/__init__.py`. It contains entries for:
- `zh-hans` (Simplified Chinese) at line 602-607
- `zh-hant` (Traditional Chinese) at line 608-613
- Various fallback entries like `zh-cn` → `zh-hans`, `zh-tw` → `zh-hant` at lines 599-628

The fallback logic that fails is in `get_language_info()` at `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/utils/translation/__init__.py:279-287`.

This bug affects any Django application using Chinese localization where users might:
- Configure their browser to send `Accept-Language: zh` headers
- Set `LANGUAGE_CODE = 'zh'` in Django settings
- Use custom Chinese language variants not explicitly listed in LANG_INFO

## Proposed Fix

Add a base Chinese language entry to LANG_INFO. Since Chinese has two main writing systems without a universal default, the base entry should follow Django's existing pattern for ambiguous cases:

```diff
--- a/django/conf/locale/__init__.py
+++ b/django/conf/locale/__init__.py
@@ -596,6 +596,9 @@ LANG_INFO = {
         "name": "Vietnamese",
         "name_local": "Tiếng Việt",
     },
+    "zh": {
+        "fallback": ["zh-hans"],
+    },
     "zh-cn": {
         "fallback": ["zh-hans"],
     },
```

This minimal fix:
- Makes `zh` a valid language code that falls back to Simplified Chinese
- Follows the same pattern as existing entries like `zh-cn`
- Restores consistency with all other languages in LANG_INFO
- Fixes the fallback mechanism for unknown Chinese variants