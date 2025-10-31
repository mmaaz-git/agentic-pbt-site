import numpy as np
from pandas.core import algorithms

# Test the exact case from the bug report
print("Test 1: Single null character")
values = ['\x00']
arr = np.array(values)

codes, uniques = algorithms.factorize(arr)

reconstructed = [uniques[code] for code in codes]

print(f"Original: {repr(values[0])}")
print(f"Reconstructed: {repr(reconstructed[0])}")
print(f"Are they equal? {values[0] == reconstructed[0]}")
print()

# Test double null characters
print("Test 2: Double null characters")
values2 = ['\x00\x00']
arr2 = np.array(values2)

codes2, uniques2 = algorithms.factorize(arr2)
reconstructed2 = [uniques2[code] for code in codes2]

print(f"Original: {repr(values2[0])}")
print(f"Reconstructed: {repr(reconstructed2[0])}")
print(f"Are they equal? {values2[0] == reconstructed2[0]}")
print()

# Test null character in the middle
print("Test 3: Null character in the middle")
values3 = ['a\x00b']
arr3 = np.array(values3)

codes3, uniques3 = algorithms.factorize(arr3)
reconstructed3 = [uniques3[code] for code in codes3]

print(f"Original: {repr(values3[0])}")
print(f"Reconstructed: {repr(reconstructed3[0])}")
print(f"Are they equal? {values3[0] == reconstructed3[0]}")
print()

# Test other control character
print("Test 4: Other control character (\\x01)")
values4 = ['\x01']
arr4 = np.array(values4)

codes4, uniques4 = algorithms.factorize(arr4)
reconstructed4 = [uniques4[code] for code in codes4]

print(f"Original: {repr(values4[0])}")
print(f"Reconstructed: {repr(reconstructed4[0])}")
print(f"Are they equal? {values4[0] == reconstructed4[0]}")
print()

# Test mixed case
print("Test 5: Mixed strings including null character")
values5 = ['hello', '\x00', 'world', '\x00\x00', 'a\x00b']
arr5 = np.array(values5)

codes5, uniques5 = algorithms.factorize(arr5)
reconstructed5 = [uniques5[code] for code in codes5]

print("Original values:")
for v in values5:
    print(f"  {repr(v)}")
print("Reconstructed values:")
for r in reconstructed5:
    print(f"  {repr(r)}")
print("All equal?")
for i, (orig, recon) in enumerate(zip(values5, reconstructed5)):
    eq = orig == recon
    print(f"  {i}: {repr(orig)} == {repr(recon)}: {eq}")