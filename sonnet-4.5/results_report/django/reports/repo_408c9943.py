#!/usr/bin/env python3
"""
Minimal reproduction script for Django Azerbaijani bidi bug.
This demonstrates that Azerbaijani (az) is incorrectly marked as bidi=True
despite using Latin script which is left-to-right.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.conf.locale import LANG_INFO

# Get Azerbaijani language info
az_info = LANG_INFO['az']

print("=== Azerbaijani Language Configuration ===")
print(f"Language code: az")
print(f"Name: {az_info['name']}")
print(f"Name (local): {az_info['name_local']}")
print(f"Marked as bidi (RTL): {az_info['bidi']}")
print()

# Show the characters in the local name
print("=== Character Analysis of 'Azərbaycanca' ===")
for i, char in enumerate(az_info['name_local']):
    print(f"  Position {i}: '{char}' (U+{ord(char):04X}) - {char.isalpha() and 'Letter' or 'Other'}")
print()

# Compare with other bidi languages
print("=== Comparison with Other Bidi Languages ===")
bidi_langs = [(code, info) for code, info in LANG_INFO.items()
              if isinstance(info, dict) and info.get('bidi', False)]

for code, info in sorted(bidi_langs):
    name_local = info.get('name_local', '')
    print(f"{code:6} | bidi={info['bidi']} | name_local='{name_local}'")
print()

# Check if Azerbaijani contains any RTL script characters
def contains_rtl_characters(text):
    """Check if text contains any RTL script characters."""
    rtl_ranges = [
        (0x0590, 0x05FF),  # Hebrew
        (0x0600, 0x06FF),  # Arabic
        (0x0700, 0x074F),  # Syriac
        (0x0750, 0x077F),  # Arabic Supplement
        (0x0780, 0x07BF),  # Thaana
        (0x07C0, 0x07FF),  # N'Ko
        (0x0800, 0x083F),  # Samaritan
        (0x0840, 0x085F),  # Mandaic
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB1D, 0xFB4F),  # Hebrew presentation forms
        (0xFB50, 0xFDFF),  # Arabic presentation forms A
        (0xFE70, 0xFEFF),  # Arabic presentation forms B
    ]

    for char in text:
        code_point = ord(char)
        for start, end in rtl_ranges:
            if start <= code_point <= end:
                return True
    return False

print("=== RTL Script Analysis ===")
for code, info in sorted(bidi_langs):
    name_local = info.get('name_local', '')
    has_rtl = contains_rtl_characters(name_local)
    status = "✓ CORRECT" if has_rtl else "✗ INCORRECT"
    print(f"{code:6} | name_local='{name_local}' | Contains RTL: {has_rtl} | {status}")

print("\n=== CONCLUSION ===")
print(f"Azerbaijani ('az') is marked as bidi=True but its name_local '{az_info['name_local']}'")
print(f"contains NO RTL script characters. This is a BUG.")
print(f"The language should be marked as bidi=False since it uses Latin script.")