import numpy as np
from pandas.core.algorithms import factorize

# Test case with empty string and null character
values = np.array(['', '\x00'], dtype=object)
codes, uniques = factorize(values)

print(f"Input: {[repr(v) for v in values]}")
print(f"Codes: {codes}")
print(f"Uniques: {[repr(v) for v in uniques]}")

# Attempt to reconstruct original values
reconstructed = uniques.take(codes)
print(f"Reconstructed: {[repr(v) for v in reconstructed]}")
print(f"Match original? {np.array_equal(reconstructed, values)}")

# Additional test cases
print("\n--- Additional test cases ---")

# Test: single null vs double null
values2 = np.array(['\x00', '\x00\x00'], dtype=object)
codes2, uniques2 = factorize(values2)
print(f"\nInput: {[repr(v) for v in values2]}")
print(f"Codes: {codes2}")
print(f"Uniques: {[repr(v) for v in uniques2]}")
print(f"Reconstructed matches? {np.array_equal(uniques2.take(codes2), values2)}")

# Test: empty, null, and regular char
values3 = np.array(['', '\x00', 'a'], dtype=object)
codes3, uniques3 = factorize(values3)
print(f"\nInput: {[repr(v) for v in values3]}")
print(f"Codes: {codes3}")
print(f"Uniques: {[repr(v) for v in uniques3]}")
print(f"Reconstructed matches? {np.array_equal(uniques3.take(codes3), values3)}")

# Test that works correctly: empty and \x01
values4 = np.array(['', '\x01'], dtype=object)
codes4, uniques4 = factorize(values4)
print(f"\nInput: {[repr(v) for v in values4]}")
print(f"Codes: {codes4}")
print(f"Uniques: {[repr(v) for v in uniques4]}")
print(f"Reconstructed matches? {np.array_equal(uniques4.take(codes4), values4)}")