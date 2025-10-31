#!/usr/bin/env python3

from scipy.constants import precision, value, physical_constants, find

# First, test the specific example from the bug report
print("=== Reproducing specific example ===")
key = 'Sackur-Tetrode constant (1 K, 100 kPa)'
constant_value = value(key)
constant_precision = precision(key)

print(f"Key: {key}")
print(f"Value: {constant_value}")
print(f"Precision: {constant_precision}")
print(f"Is precision negative?: {constant_precision < 0}")

raw = physical_constants[key]
print(f"Raw data: {raw}")

# Now test all constants to find how many have negative precision
print("\n=== Testing all constants for negative precision ===")
all_keys = find(None, disp=False)
negative_precision_constants = []

for key in all_keys:
    try:
        prec = precision(key)
        if prec < 0:
            negative_precision_constants.append((key, prec))
    except:
        pass

print(f"Total constants checked: {len(all_keys)}")
print(f"Constants with negative precision: {len(negative_precision_constants)}")

if negative_precision_constants:
    print("\nFirst 5 examples of constants with negative precision:")
    for i, (k, p) in enumerate(negative_precision_constants[:5]):
        val = value(k)
        print(f"  {i+1}. {k}")
        print(f"     Value: {val}, Precision: {p}")

# Run the property-based test from the bug report
print("\n=== Running property-based test ===")
def test_precision_is_non_negative():
    all_keys = find(None, disp=False)
    failures = []
    for key in all_keys:
        prec = precision(key)
        if prec < 0:
            failures.append(f"precision('{key}') = {prec}, should be non-negative")
    return failures

failures = test_precision_is_non_negative()
if failures:
    print(f"Test FAILED with {len(failures)} failures")
    print("First failure:", failures[0])
else:
    print("Test PASSED - all precisions are non-negative")