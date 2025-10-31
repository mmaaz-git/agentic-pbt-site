from django.utils.datastructures import CaseInsensitiveMapping

# Test case 1: German eszett (found by Hypothesis)
print("Test case 1: German eszett 'ß'")
mapping = CaseInsensitiveMapping({'ß': 0})
print(f"Access with 'ß': {mapping['ß']}")

try:
    print(f"'ß'.upper() = '{('ß'.upper())}'")
    print(f"Access with 'SS' (uppercase): {mapping['SS']}")
except KeyError as e:
    print(f"KeyError when accessing with uppercase: {e}")

print(f"'ß'.lower() = '{('ß'.lower())}' (stays as 'ß')")
print(f"'ß'.upper() = '{('ß'.upper())}' (becomes 'SS')")
print(f"'SS'.lower() = '{('SS'.lower())}' (becomes 'ss')")
print(f"Key stored internally: 'ß'.lower() = 'ß'")
print(f"Key looked up: 'SS'.lower() = 'ss' (mismatch!)")

print("\n" + "="*50 + "\n")

# Test case 2: Micro sign (from original report)
print("Test case 2: Micro sign 'µ'")
mapping2 = CaseInsensitiveMapping({'µ': 0})
print(f"Access with 'µ' (MICRO SIGN): {mapping2['µ']}")

try:
    print(f"'µ'.upper() = '{('µ'.upper())}'")
    print(f"Access with 'Μ' (uppercase): {mapping2['Μ']}")
except KeyError as e:
    print(f"KeyError when accessing with uppercase: {e}")

print(f"'µ' (MICRO SIGN) = U+{ord('µ'):04X}")
print(f"'µ'.lower() = '{('µ'.lower())}' = U+{ord('µ'.lower()):04X}")
print(f"'µ'.upper() = '{('µ'.upper())}' = U+{ord('µ'.upper()):04X}")
print(f"'µ'.upper().lower() = '{('µ'.upper().lower())}' = U+{ord('µ'.upper().lower()):04X}")