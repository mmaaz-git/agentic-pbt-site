#!/usr/bin/env python3
"""Minimal reproduction case for scipy.constants.precision() returning negative values."""

from scipy.constants import precision, value, physical_constants

# Test with the Sackur-Tetrode constant which has a negative value
key = 'Sackur-Tetrode constant (1 K, 100 kPa)'

# Get the value and precision
constant_value = value(key)
constant_precision = precision(key)

print(f"Constant: {key}")
print(f"Value: {constant_value}")
print(f"Precision: {constant_precision}")
print(f"Is precision negative? {constant_precision < 0}")

# Show the raw data from physical_constants
raw = physical_constants[key]
print(f"\nRaw data from physical_constants:")
print(f"  Value: {raw[0]}")
print(f"  Unit: {raw[1]}")
print(f"  Uncertainty: {raw[2]}")

# Demonstrate the calculation
calculated_precision = raw[2] / raw[0]
print(f"\nCalculated precision (uncertainty/value): {calculated_precision}")
print(f"This matches precision() output: {calculated_precision == constant_precision}")

# What it should be according to metrological standards
correct_precision = abs(raw[2] / raw[0])
print(f"\nCorrect precision (|uncertainty/value|): {correct_precision}")
print(f"This would be non-negative: {correct_precision >= 0}")