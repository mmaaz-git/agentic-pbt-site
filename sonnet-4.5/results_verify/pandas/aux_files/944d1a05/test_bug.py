import numpy as np
from pandas.core.util.hashing import hash_array

# Test the specific failing case from the bug report
arr = np.array([''], dtype=object)
hash_key = '000000000000000\x80'

print(f"hash_key: {repr(hash_key)}")
print(f"hash_key length (characters): {len(hash_key)}")
print(f"hash_key encoded length (bytes): {len(hash_key.encode('utf8'))}")
print(f"hash_key encoded: {hash_key.encode('utf8')}")

try:
    result = hash_array(arr, hash_key=hash_key)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test with a valid 16-byte key
print("\n--- Testing with valid 16-byte key ---")
valid_key = '0123456789123456'
print(f"valid_key: {repr(valid_key)}")
print(f"valid_key length (characters): {len(valid_key)}")
print(f"valid_key encoded length (bytes): {len(valid_key.encode('utf8'))}")

result = hash_array(arr, hash_key=valid_key)
print(f"Result: {result}")

# Test with another multi-byte character
print("\n--- Testing with another multi-byte character ---")
multibyte_key = '000000000000000ñ'
print(f"multibyte_key: {repr(multibyte_key)}")
print(f"multibyte_key length (characters): {len(multibyte_key)}")
print(f"multibyte_key encoded length (bytes): {len(multibyte_key.encode('utf8'))}")
print(f"multibyte_key encoded: {multibyte_key.encode('utf8')}")

try:
    result = hash_array(arr, hash_key=multibyte_key)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")