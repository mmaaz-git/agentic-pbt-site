#!/usr/bin/env python3
"""Verify the mathematical claims in the bug report"""

import decimal
import math

# Set high precision for decimal calculations
decimal.getcontext().prec = 50

def analyze_number(str_value):
    """Analyze a number string in detail"""
    print(f"\nAnalyzing: {str_value}")
    print("=" * 50)

    # Parse as float
    float_val = float(str_value)
    print(f"As Python float: {float_val}")
    print(f"Scientific notation: {float_val:.30e}")

    # Parse as Decimal for exact representation
    decimal_val = decimal.Decimal(str_value)
    print(f"As Decimal (exact): {decimal_val}")

    # Check the formatting behavior
    formatted_20f = "%0.20f" % float_val
    print(f"Formatted with %0.20f: '{formatted_20f}'")

    stripped = formatted_20f.rstrip('0').rstrip('.')
    print(f"After stripping zeros: '{stripped}'")

    # Check if we lose precision
    if stripped == '0' and float_val != 0:
        print("*** PRECISION LOST: Non-zero value becomes '0' ***")
        return False
    else:
        # Check if we can recover the original
        recovered = float(stripped) if stripped and stripped != '.' else 0.0
        print(f"Recovered float: {recovered}")
        if recovered == float_val:
            print("✓ Round-trip successful")
            return True
        else:
            print(f"✗ Round-trip failed: {recovered} != {float_val}")
            return False

# Test the specific case from the bug report
print("TESTING BUG REPORT CLAIM")
print("The bug report states that 0.000000000000000000001 = 1e-21")
test_val = "0.000000000000000000001"
decimal_val = decimal.Decimal(test_val)
print(f"Decimal value: {decimal_val}")
print(f"Scientific: {decimal_val:.5e}")
float_val = float(test_val)
print(f"Float value: {float_val}")
print(f"Float in scientific: {float_val:.30e}")

# The claim is this is 1e-21
print(f"\nIs {test_val} equal to 1e-21?")
print(f"  1e-21 = {1e-21}")
print(f"  float('{test_val}') = {float(test_val)}")
print(f"  Are they equal? {float(test_val) == 1e-21}")

# Test boundary cases around 20 decimal places
print("\n\nBOUNDARY ANALYSIS: Testing around 20 decimal places")
print("-" * 50)

test_cases = [
    ("19 decimals", "0.0000000000000000001"),  # 1e-19
    ("20 decimals", "0.00000000000000000001"), # 1e-20
    ("21 decimals", "0.000000000000000000001"), # 1e-21
    ("22 decimals", "0.0000000000000000000001"), # 1e-22
]

results = []
for label, num_str in test_cases:
    print(f"\n{label}: {num_str}")
    success = analyze_number(num_str)
    results.append((label, success))

print("\n\nSUMMARY")
print("-" * 50)
for label, success in results:
    status = "✓ PASSES" if success else "✗ FAILS"
    print(f"{label}: {status}")

# Check what numbers can be represented exactly
print("\n\nPRECISION LIMITS OF %0.20f FORMAT")
print("-" * 50)
print("The %0.20f format can show up to 20 decimal places")
print("Numbers smaller than 1e-20 will round to 0.00000000000000000000")
print("After stripping trailing zeros, this becomes '0'")

# Demonstrate
smallest_representable = 1e-20
smaller = 1e-21

print(f"\n1e-20: '%0.20f' % {smallest_representable} = '{('%0.20f' % smallest_representable)}'")
print(f"After strip: '{('%0.20f' % smallest_representable).rstrip('0').rstrip('.')}'")

print(f"\n1e-21: '%0.20f' % {smaller} = '{('%0.20f' % smaller)}'")
print(f"After strip: '{('%0.20f' % smaller).rstrip('0').rstrip('.')}'")

print("\nConclusion: Any decimal number with more than 20 decimal places")
print("(i.e., smaller than 1e-20) will fail the round-trip test.")