import numpy as np
import pandas as pd

# Create complex arrays with the same values but different dtypes
arr64 = np.array([1+2j, 3+4j], dtype=np.complex64)
arr128 = np.array([1+2j, 3+4j], dtype=np.complex128)

# Hash both arrays
hash64 = pd.util.hash_array(arr64)
hash128 = pd.util.hash_array(arr128)

# Calculate what the hash64 SHOULD be if it used the same formula as hash128
hash64_real = pd.util.hash_array(arr64.real)
hash64_imag = pd.util.hash_array(arr64.imag)
expected64 = hash64_real + 23 * hash64_imag

print("Complex64 array:", arr64)
print(f"complex64 hash:  {hash64}")
print(f"Expected hash:   {expected64}")
print(f"Match: {np.array_equal(hash64, expected64)}")

print("\nComplex128 array:", arr128)
print(f"complex128 hash: {hash128}")

# Verify complex128 uses the expected formula
hash128_real = pd.util.hash_array(arr128.real)
hash128_imag = pd.util.hash_array(arr128.imag)
expected128 = hash128_real + 23 * hash128_imag
print(f"Expected hash:   {expected128}")
print(f"Match: {np.array_equal(hash128, expected128)}")

print("\nThe bug: complex64 and complex128 arrays with identical values hash differently!")
print(f"Same values? {np.array_equal(arr64, arr128)}")
print(f"Same hashes? {np.array_equal(hash64, hash128)}")