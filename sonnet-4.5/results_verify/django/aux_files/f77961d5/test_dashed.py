#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.utils.translation import to_locale

# Test with dashed inputs (expected behavior)
print("Testing with dashed inputs (expected behavior):")
print(f"to_locale('en-us') = {repr(to_locale('en-us'))}")
print(f"to_locale('EN-US') = {repr(to_locale('EN-US'))}")
print(f"to_locale('fr-ca') = {repr(to_locale('fr-ca'))}")
print(f"to_locale('FR-CA') = {repr(to_locale('FR-CA'))}")
print(f"to_locale('sr-latn') = {repr(to_locale('sr-latn'))}")
print(f"to_locale('SR-LATN') = {repr(to_locale('SR-LATN'))}")

# Test edge cases
print("\nTesting edge cases:")
print(f"to_locale('en') = {repr(to_locale('en'))}")
print(f"to_locale('EN') = {repr(to_locale('EN'))}")
print(f"to_locale('ENU') = {repr(to_locale('ENU'))}")
print(f"to_locale('ABC') = {repr(to_locale('ABC'))}")
print(f"to_locale('ABCD') = {repr(to_locale('ABCD'))}")
print(f"to_locale('ABCDEF') = {repr(to_locale('ABCDEF'))}")