#!/usr/bin/env python3
"""Minimal reproduction case for django.utils.translation.to_locale bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.utils.translation import to_locale

# Test cases showing the bug
test_cases = [
    'GerMAN',
    'EnGlish',
    'FrEnch',
    'UPPERCASE',
    'MiXeDcAsE',
    'lower',
    'en',
    'EN',
    'fr-FR',
    'en-US',
    'EN-US',
    'de-DE-1996'
]

print("Testing to_locale() function:")
print("=" * 50)

for test in test_cases:
    result = to_locale(test)
    print(f"to_locale('{test:15}') = '{result}'")

print("\n" + "=" * 50)
print("\nExpected behavior for non-hyphenated inputs:")
print("The entire string should be lowercase since line 233")
print("computes: lang = language.lower().partition('-')[0]")
print("But line 235 incorrectly uses: language[:3].lower() + language[3:]")
print("\nThis causes inconsistent casing like 'gerMAN' from 'GerMAN'")