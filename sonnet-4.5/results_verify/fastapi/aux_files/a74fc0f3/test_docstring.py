#!/usr/bin/env python3
from decimal import Decimal
from fastapi.encoders import decimal_encoder

# Test the examples from the docstring
print("Testing docstring examples:")
print(f"decimal_encoder(Decimal('1.0')) = {decimal_encoder(Decimal('1.0'))}")
print(f"decimal_encoder(Decimal('1')) = {decimal_encoder(Decimal('1'))}")

# Check what the docstring claims about ConstrainedDecimal and Numeric(x,0)
print("\nTesting integer decimals (no fractional part):")
test_integers = [
    Decimal('1'),
    Decimal('100'),
    Decimal('1000000'),
    Decimal('1.00'),  # No fractional part, just trailing zeros
    Decimal('100.0'),
]

for d in test_integers:
    encoded = decimal_encoder(d)
    decoded = Decimal(str(encoded))
    print(f"{d} -> {encoded} (type: {type(encoded).__name__}) -> {decoded}: {'✓' if decoded == d else '✗'}")

print("\nTesting fractional decimals:")
test_fractions = [
    Decimal('1.1'),
    Decimal('1.01'),
    Decimal('1.001'),
    Decimal('1.0001'),
    Decimal('1.00001'),
    Decimal('1.000001'),
    Decimal('1.0000001'),
    Decimal('1.00000001'),
    Decimal('1.000000001'),
]

for d in test_fractions:
    encoded = decimal_encoder(d)
    decoded = Decimal(str(encoded))
    print(f"{d} -> {encoded} -> {decoded}: {'✓' if decoded == d else '✗'}")

# Check the exponent logic
print("\nUnderstanding the exponent logic:")
for val_str in ['1', '1.0', '1.00', '1.1', '0.1', '10', '100']:
    d = Decimal(val_str)
    exp = d.as_tuple().exponent
    encoded = decimal_encoder(d)
    print(f"Decimal('{val_str}').as_tuple().exponent = {exp}, encoded as {encoded} ({type(encoded).__name__})")