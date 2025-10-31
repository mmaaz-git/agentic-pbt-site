#!/usr/bin/env python3

# Test Python's Unicode handling for ß
s = 'ß'

print("Testing German ß character:")
print(f"Original: {repr(s)} (length {len(s)})")
print(f"s.upper(): {repr(s.upper())} (length {len(s.upper())})")
print(f"s.capitalize(): {repr(s.capitalize())} (length {len(s.capitalize())})")
print(f"s[0]: {repr(s[0])}")
print(f"s[0].upper(): {repr(s[0].upper())} (length {len(s[0].upper())})")
print(f"s[:1]: {repr(s[:1])}")
print(f"s[:1].upper(): {repr(s[:1].upper())} (length {len(s[:1].upper())})")

print("\nTesting with 'ßeta':")
s2 = 'ßeta'
print(f"Original: {repr(s2)}")
print(f"s2.upper(): {repr(s2.upper())}")
print(f"s2.capitalize(): {repr(s2.capitalize())}")
print(f"s2[:1].upper() + s2[1:]: {repr(s2[:1].upper() + s2[1:])}")

print("\nThis is standard Unicode behavior:")
print("German ß (LATIN SMALL LETTER SHARP S, U+00DF)")
print("uppercases to SS (two separate S characters) per Unicode standard")