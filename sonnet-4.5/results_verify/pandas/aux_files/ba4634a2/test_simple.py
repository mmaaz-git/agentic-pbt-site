import numpy as np
import pandas.arrays as pa

# Test the simple reproducer
print("Testing simple reproducer with IntegerArray:")
arr = pa.IntegerArray(np.array([0], dtype='int64'), mask=np.array([True]))
print(f"Array: {arr}")
print(f"Array data: {arr._data}")
print(f"Array mask: {arr._mask}")

codes, uniques = arr.factorize()
print(f"Codes: {codes}")
print(f"Uniques: {uniques}")
print(f"Uniques length: {len(uniques)}")

try:
    reconstructed = uniques[codes]
    print(f"Reconstructed: {reconstructed}")
except IndexError as e:
    print(f"IndexError: {e}")

# Test with use_na_sentinel=False
print("\n\nTesting with use_na_sentinel=False:")
codes2, uniques2 = arr.factorize(use_na_sentinel=False)
print(f"Codes: {codes2}")
print(f"Uniques: {uniques2}")
print(f"Uniques length: {len(uniques2)}")

try:
    reconstructed2 = uniques2[codes2]
    print(f"Reconstructed: {reconstructed2}")
except IndexError as e:
    print(f"IndexError: {e}")