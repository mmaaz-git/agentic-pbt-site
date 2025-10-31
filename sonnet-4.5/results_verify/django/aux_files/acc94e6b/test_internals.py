#!/usr/bin/env python3
from decimal import Decimal

# Test internal representations
test_cases = [
    Decimal("0"),
    Decimal("0.0"),
    Decimal("0.00"),
    Decimal("1"),
    Decimal("1.0"),
    Decimal("1.00"),
]

print("Understanding Decimal internal representations:")
print("=" * 60)
for d in test_cases:
    tuple_repr = d.as_tuple()
    print(f"Decimal('{d}'):")
    print(f"  Full tuple: {tuple_repr}")
    print(f"  digit_tuple: {tuple_repr[1]}")
    print(f"  exponent: {tuple_repr[2]}")
    print(f"  Value == 0: {d == 0}")
    print()