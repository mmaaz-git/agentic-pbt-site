import pandas.core.algorithms as algorithms
import numpy as np

# Test case demonstrating the bug
values = ['', '\x00']
print(f"Input values: {values}")
print(f"Input values (repr): {repr(values)}")
print()

# Call factorize
codes, uniques = algorithms.factorize(values)

print(f"Codes returned: {codes}")
print(f"Codes type: {type(codes)}")
print()

print(f"Uniques returned: {list(uniques)}")
print(f"Uniques (repr): {repr(list(uniques))}")
print(f"Uniques type: {type(uniques)}")
print()

# Reconstruct using the documented property
reconstructed = uniques.take(codes)
print(f"Reconstructed values: {list(reconstructed)}")
print(f"Reconstructed (repr): {repr(list(reconstructed))}")
print()

# Verify the round-trip property
print("Checking round-trip property...")
print(f"Original value at index 0: {repr(values[0])}")
print(f"Reconstructed at index 0: {repr(reconstructed[0])}")
print(f"Match at index 0: {values[0] == reconstructed[0]}")
print()

print(f"Original value at index 1: {repr(values[1])}")
print(f"Reconstructed at index 1: {repr(reconstructed[1])}")
print(f"Match at index 1: {values[1] == reconstructed[1]}")
print()

# The assertion that should pass according to documentation
try:
    assert values[1] == '\x00', "Original value should be null character"
    print("✓ Original value is null character")
except AssertionError as e:
    print(f"✗ Assertion failed: {e}")

try:
    assert reconstructed[1] == '\x00', "Round-trip should preserve null character"
    print("✓ Round-trip preserved null character")
except AssertionError as e:
    print(f"✗ Assertion failed: {e}")

print()
print("CONCLUSION: The factorize function violates its documented guarantee that")
print("'uniques.take(codes)' will have the same values as 'values' when input")
print("contains null characters.")