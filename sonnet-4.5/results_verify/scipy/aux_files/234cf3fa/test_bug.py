#!/usr/bin/env python3
import math
import sys
from hypothesis import given, strategies as st
from scipy.constants import convert_temperature

# First, test the hypothesis test
print("Testing the Hypothesis property-based test...")
@given(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False))
def test_same_scale_returns_same_value(temp):
    result = convert_temperature(temp, 'Celsius', 'Celsius')
    assert math.isclose(result, temp, rel_tol=1e-10)

# Test with the specific failing input mentioned
try:
    temp = 1.1754943508222875e-38
    result = convert_temperature(temp, 'Celsius', 'Celsius')
    print(f"Specific failing input test:")
    print(f"  Input: {temp}")
    print(f"  Result: {result}")
    print(f"  Expected: {temp}")
    print(f"  Are they close? {math.isclose(result, temp, rel_tol=1e-10)}")
    print()
except Exception as e:
    print(f"Error with specific input: {e}")
    print()

# Now reproduce the examples from the bug report
print("Reproducing bug report examples:")
print("-" * 40)

temp = 1e-10
result = convert_temperature(temp, 'Celsius', 'Celsius')
print(f"Input: {temp}, Result: {result}, Expected: {temp}")
print(f"  Difference: {abs(result - temp)}")
print(f"  Relative error: {abs(result - temp) / temp if temp != 0 else 'N/A'}")
print()

temp2 = 1e-20
result2 = convert_temperature(temp2, 'Celsius', 'Celsius')
print(f"Input: {temp2}, Result: {result2}, Expected: {temp2}")
print(f"  Difference: {abs(result2 - temp2)}")
print(f"  Relative error: {abs(result2 - temp2) / temp2 if temp2 != 0 else 'N/A'}")
print()

# Test with extremely small values
print("Testing with various small values:")
print("-" * 40)
test_values = [1e-5, 1e-10, 1e-15, 1e-20, 1e-30, 1e-38, 1e-50]
for val in test_values:
    result = convert_temperature(val, 'Celsius', 'Celsius')
    print(f"Input: {val:e}, Result: {result:e}, Lost precision: {result == 0.0}")

print()

# Test different scale combinations for identity
print("Testing identity conversions for all scales:")
print("-" * 40)
scales = ['Celsius', 'Kelvin', 'Fahrenheit', 'Rankine']
test_val = 1e-20
for scale in scales:
    result = convert_temperature(test_val, scale, scale)
    print(f"{scale} -> {scale}: Input={test_val:e}, Result={result:e}, Equal={result == test_val}")

print()

# Verify the math manually for small Celsius values
print("Manual verification of the math:")
print("-" * 40)
import numpy as np
test_val = 1e-20
print(f"Original value: {test_val}")
# Simulate what the function does
step1 = test_val + 273.15  # Celsius to Kelvin
print(f"After adding 273.15: {step1}")
step2 = step1 - 273.15      # Kelvin back to Celsius
print(f"After subtracting 273.15: {step2}")
print(f"Lost precision: {step2 == 0.0}")
print()

# Check floating point representation limits
print("Checking floating-point limits:")
print("-" * 40)
val = 1e-20
sum_val = val + 273.15
print(f"Can {val} + 273.15 preserve the small value?")
print(f"  273.15 in binary has ~50 bits of precision")
print(f"  {val} is about 2^{math.log2(val):.1f}")
print(f"  Difference in magnitude: ~{math.log2(273.15/val):.1f} bits")
print(f"  Result of addition: {sum_val}")
print(f"  Exactly equals 273.15? {sum_val == 273.15}")