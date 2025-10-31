#!/usr/bin/env python3
"""Minimal reproduction case for CaseInsensitiveMapping bug with German ß character."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.utils.datastructures import CaseInsensitiveMapping

# Create a CaseInsensitiveMapping with German eszett character
cim = CaseInsensitiveMapping({'ß': 'value'})

# Test retrieving with the original key
print("cim.get('ß'):", cim.get('ß'))

# Test retrieving with uppercase version (SS)
print("cim.get('SS'):", cim.get('SS'))

# Try direct access with uppercase (this will raise an error)
try:
    result = cim['SS']
    print("cim['SS']:", result)
except KeyError as e:
    print(f"cim['SS'] raised KeyError: {e}")

# Show the case transformation issue
print("\nCase transformations:")
print(f"'ß'.upper() = '{('ß').upper()}'")
print(f"'ß'.lower() = '{('ß').lower()}'")
print(f"'SS'.lower() = '{('SS').lower()}'")
print(f"'ss'.lower() = '{('ss').lower()}'")

print("\nUsing casefold() (proper Unicode case-insensitive matching):")
print(f"'ß'.casefold() = '{('ß').casefold()}'")
print(f"'SS'.casefold() = '{('SS').casefold()}'")
print(f"'ss'.casefold() = '{('ss').casefold()}'")