#!/usr/bin/env python3
"""Test to understand float precision issues"""

# Test float precision with large integers
value = 10**16 + 1
multiple = 5

print(f"Value: {value}")
print(f"Multiple: {multiple}")
print(f"Integer modulo: {value % multiple}")

# What happens when we convert to float?
float_value = float(value)
float_multiple = float(multiple)

print(f"\nFloat value: {float_value}")
print(f"Float value == original value: {float_value == value}")
print(f"Float modulo calculation: {float_value / float_multiple % 1}")

# Test with the specific failing value from bug report
value2 = 17608513714555794
print(f"\n\nValue2: {value2}")
print(f"Value2 % 5: {value2 % 5}")
print(f"Float(value2) / 5.0 % 1: {float(value2) / 5.0 % 1}")

# Check if float can represent these large integers exactly
import sys
print(f"\n\nMax exact integer in float: 2^53 = {2**53}")
print(f"Test value 10^16 + 1 = {10**16 + 1}")
print(f"10^16 + 1 > 2^53: {(10**16 + 1) > 2**53}")

# Show that float loses precision
print(f"\n\nPrecision loss demonstration:")
for i in range(5):
    test_val = 10**16 + i
    float_val = float(test_val)
    print(f"int({test_val}) -> float({float_val}) -> int({int(float_val)})")