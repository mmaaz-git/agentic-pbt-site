import numpy as np
from pandas.core.util.hashing import hash_array

c64 = np.array([1j], dtype=np.complex64)
c128 = np.array([1j], dtype=np.complex128)

hash_c64 = hash_array(c64)
hash_c128 = hash_array(c128)

print(f"Complex64 hash: {hash_c64[0]}")
print(f"Complex128 hash: {hash_c128[0]}")

hash_real_c64 = hash_array(c64.real)
hash_imag_c64 = hash_array(c64.imag)
expected_c64 = hash_real_c64 + 23 * hash_imag_c64

print(f"\nExpected hash for c64 (using formula): {expected_c64[0]}")
print(f"Actual hash for c64: {hash_c64[0]}")
print(f"Match: {hash_c64[0] == expected_c64[0]}")

# Let's also check the behavior with complex128
hash_real_c128 = hash_array(c128.real)
hash_imag_c128 = hash_array(c128.imag)
expected_c128 = hash_real_c128 + 23 * hash_imag_c128

print(f"\nExpected hash for c128 (using formula): {expected_c128[0]}")
print(f"Actual hash for c128: {hash_c128[0]}")
print(f"Match for c128: {hash_c128[0] == expected_c128[0]}")