#!/usr/bin/env python3
"""Minimal reproduction case for Django i18n GetLanguageInfoListNode bug"""

import os
import sys
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

import django
from django.conf import settings
settings.configure(
    DEBUG=True,
    LANGUAGE_CODE='en-us',
    USE_I18N=True,
    USE_L10N=True,
)
django.setup()

from django.templatetags.i18n import GetLanguageInfoListNode
from django.utils import translation

# Create a node instance
node = GetLanguageInfoListNode(None, 'result')

print("=" * 60)
print("Testing GetLanguageInfoListNode.get_language_info()")
print("=" * 60)

# Test case 1: Sequence with single-character first element (FAILS)
print("\nTest 1: Sequence with single-character first element")
print("-" * 50)
language = ['x', 'Unknown Language']
print(f"Input: {language}")
print(f"Type: {type(language)}")
print(f"language[0] = {language[0]!r} (type: {type(language[0])})")
print(f"len(language[0]) = {len(language[0])}")

try:
    # Trace through the logic
    if len(language[0]) > 1:
        print("Logic: len(language[0]) > 1 is TRUE")
        print(f"Would call: translation.get_language_info({language[0]!r})")
    else:
        print("Logic: len(language[0]) > 1 is FALSE")
        print(f"Would call: translation.get_language_info(str({language!r}))")
        print(f"         = translation.get_language_info({str(language)!r})")

    # Actually call the method
    result = node.get_language_info(language)
    print(f"SUCCESS: Result = {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# Test case 2: Sequence with multi-character first element (WORKS)
print("\n" + "=" * 60)
print("Test 2: Sequence with multi-character first element")
print("-" * 50)
language = ['en', 'English']
print(f"Input: {language}")
print(f"Type: {type(language)}")
print(f"language[0] = {language[0]!r} (type: {type(language[0])})")
print(f"len(language[0]) = {len(language[0])}")

try:
    # Trace through the logic
    if len(language[0]) > 1:
        print("Logic: len(language[0]) > 1 is TRUE")
        print(f"Would call: translation.get_language_info({language[0]!r})")
    else:
        print("Logic: len(language[0]) > 1 is FALSE")
        print(f"Would call: translation.get_language_info(str({language!r}))")
        print(f"         = translation.get_language_info({str(language)!r})")

    # Actually call the method
    result = node.get_language_info(language)
    print(f"SUCCESS: Result = {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# Test case 3: String language code (WORKS by coincidence)
print("\n" + "=" * 60)
print("Test 3: String language code")
print("-" * 50)
language = 'en'
print(f"Input: {language!r}")
print(f"Type: {type(language)}")
print(f"language[0] = {language[0]!r} (type: {type(language[0])})")
print(f"len(language[0]) = {len(language[0])}")

try:
    # Trace through the logic
    if len(language[0]) > 1:
        print("Logic: len(language[0]) > 1 is TRUE")
        print(f"Would call: translation.get_language_info({language[0]!r})")
    else:
        print("Logic: len(language[0]) > 1 is FALSE")
        print(f"Would call: translation.get_language_info(str({language!r}))")
        print(f"         = translation.get_language_info({str(language)!r})")

    # Actually call the method
    result = node.get_language_info(language)
    print(f"SUCCESS: Result = {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)
print("""
The bug occurs because the code uses len(language[0]) > 1 to distinguish
between strings and sequences. This fails for sequences with single-character
first elements:

- For ['x', 'Unknown']: language[0] = 'x', len('x') = 1, so it calls
  translation.get_language_info(str(['x', 'Unknown'])) which is
  translation.get_language_info("['x', 'Unknown']") - INVALID!

- For ['en', 'English']: language[0] = 'en', len('en') = 2, so it calls
  translation.get_language_info('en') - CORRECT!

- For 'en': language[0] = 'e', len('e') = 1, so it calls
  translation.get_language_info(str('en')) which is
  translation.get_language_info('en') - WORKS BY COINCIDENCE!

The fix is to use isinstance(language, str) to properly check the type.
""")