#!/usr/bin/env python3
from decimal import Decimal
from fastapi.encoders import decimal_encoder

# Test with the specific failing value
failing_value = Decimal('9202420.974752872')

print(f"Original decimal: {failing_value}")
encoded = decimal_encoder(failing_value)
print(f"Encoded value: {encoded}")
print(f"Type of encoded: {type(encoded)}")

decoded = Decimal(str(encoded))
print(f"Decoded value: {decoded}")

print(f"Round-trip preserved? {decoded == failing_value}")
print(f"Loss of precision: {failing_value - decoded}")

# Test with more examples
print("\n--- Testing more examples ---")
test_values = [
    Decimal('1.1'),
    Decimal('0.333333333333333333'),
    Decimal('999999999999.999999999'),
    Decimal('0.1'),
    Decimal('0.2'),
    Decimal('0.3'),
]

for val in test_values:
    enc = decimal_encoder(val)
    dec = Decimal(str(enc))
    matches = dec == val
    print(f"{val} -> {enc} -> {dec}: {'✓' if matches else f'✗ (diff: {val - dec})'}")