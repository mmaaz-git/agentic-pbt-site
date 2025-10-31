from django.utils.translation import to_locale

# Test cases showing the bug
print("Testing to_locale() with uppercase strings without dashes:")
print(f"to_locale('AAAA') = {to_locale('AAAA')!r}")
print(f"to_locale('ENUS') = {to_locale('ENUS')!r}")
print(f"to_locale('FRCA') = {to_locale('FRCA')!r}")
print(f"to_locale('DEDE') = {to_locale('DEDE')!r}")
print(f"to_locale('ABCDEFGH') = {to_locale('ABCDEFGH')!r}")

print("\nFor comparison, strings with 3 or fewer characters:")
print(f"to_locale('EN') = {to_locale('EN')!r}")
print(f"to_locale('FRA') = {to_locale('FRA')!r}")

print("\nExpected behavior (what should happen):")
print("All characters should be lowercase when no dash is present")
print("Expected: 'aaaa', 'enus', 'frca', 'dede', 'abcdefgh', 'en', 'fra'")

print("\nActual behavior shows only first 3 chars are lowercased")