#!/usr/bin/env python3
from decimal import Decimal
from fastapi.encoders import decimal_encoder
import sys

print("Testing precision loss in decimal_encoder:")
print("=" * 60)

# Test various precision levels
test_cases = [
    # Small numbers with high precision
    Decimal('0.123456789012345678901234567890'),
    Decimal('0.999999999999999999999999999999'),

    # Large numbers with fractional parts
    Decimal('9202420.974752872'),  # The failing case from bug report
    Decimal('123456789.123456789'),
    Decimal('999999999999.999999999999'),

    # Edge cases around float precision limits (53 bits mantissa)
    Decimal('9007199254740992.1'),  # 2^53 + 0.1
    Decimal('9007199254740993.1'),  # 2^53 + 1 + 0.1

    # Scientific notation
    Decimal('1.23456789E10'),
    Decimal('1.23456789E-10'),
]

failures = []
for dec_val in test_cases:
    encoded = decimal_encoder(dec_val)
    decoded = Decimal(str(encoded))
    matches = decoded == dec_val

    if not matches:
        failures.append({
            'original': dec_val,
            'encoded': encoded,
            'decoded': decoded,
            'diff': dec_val - decoded
        })
        print(f"✗ {dec_val}")
        print(f"  Encoded: {encoded}")
        print(f"  Decoded: {decoded}")
        print(f"  Loss: {dec_val - decoded}")
    else:
        print(f"✓ {dec_val} -> {encoded}")

print("\n" + "=" * 60)
print(f"Summary: {len(failures)}/{len(test_cases)} cases lost precision")

if failures:
    print("\nFloat representation limits:")
    print(f"sys.float_info.dig = {sys.float_info.dig} (decimal digits of precision)")
    print(f"sys.float_info.mant_dig = {sys.float_info.mant_dig} (mantissa bits)")
    print("\nThe issue: Converting Decimal to float loses precision beyond ~15-17 decimal digits")