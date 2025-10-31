from django.utils.datastructures import CaseInsensitiveMapping

# Create a CaseInsensitiveMapping with the MICRO SIGN character
mapping = CaseInsensitiveMapping({'µ': 0})

# Access with the original key (µ - MICRO SIGN U+00B5)
print("Access with 'µ' (MICRO SIGN):", mapping['µ'])

# Try to access with uppercase (Μ - GREEK CAPITAL LETTER MU U+039C)
try:
    print("Access with 'µ'.upper() =", repr('µ'.upper()))
    print("Access with 'Μ' (uppercase):", mapping['Μ'])
except KeyError as e:
    print(f"KeyError when accessing with uppercase: {e}")

# Show the Unicode transformation chain
print("\nUnicode transformation chain:")
print(f"'µ' (MICRO SIGN) = U+{ord('µ'):04X}")
print(f"'µ'.lower() = '{('µ'.lower())}' = U+{ord('µ'.lower()):04X}")
print(f"'µ'.upper() = '{('µ'.upper())}' = U+{ord('µ'.upper()):04X}")
print(f"'µ'.upper().lower() = '{('µ'.upper().lower())}' = U+{ord('µ'.upper().lower()):04X}")
print(f"'µ'.lower() == 'µ'.upper().lower(): {('µ'.lower() == 'µ'.upper().lower())}")