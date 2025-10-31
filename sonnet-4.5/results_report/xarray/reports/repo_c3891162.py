import numpy as np
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')
from xarray.coding.strings import encode_string_array, decode_bytes_array

# Test case 1: Single null byte
original = np.array(['\x00'], dtype=object)
print(f"Test 1: Single null byte")
print(f"Original: {original!r}")
encoded = encode_string_array(original)
print(f"Encoded: {encoded!r}")
decoded = decode_bytes_array(encoded)
print(f"Decoded: {decoded!r}")
print(f"Round-trip successful: {np.array_equal(decoded, original)}")
print()

# Test case 2: Null byte at beginning
original2 = np.array(['\x00hello'], dtype=object)
print(f"Test 2: Null byte at beginning")
print(f"Original: {original2!r}")
encoded2 = encode_string_array(original2)
print(f"Encoded: {encoded2!r}")
decoded2 = decode_bytes_array(encoded2)
print(f"Decoded: {decoded2!r}")
print(f"Round-trip successful: {np.array_equal(decoded2, original2)}")
print()

# Test case 3: Null byte in middle
original3 = np.array(['a\x00b'], dtype=object)
print(f"Test 3: Null byte in middle")
print(f"Original: {original3!r}")
encoded3 = encode_string_array(original3)
print(f"Encoded: {encoded3!r}")
decoded3 = decode_bytes_array(encoded3)
print(f"Decoded: {decoded3!r}")
print(f"Round-trip successful: {np.array_equal(decoded3, original3)}")
print()

# Test case 4: Null byte at end
original4 = np.array(['hello\x00'], dtype=object)
print(f"Test 4: Null byte at end")
print(f"Original: {original4!r}")
encoded4 = encode_string_array(original4)
print(f"Encoded: {encoded4!r}")
decoded4 = decode_bytes_array(encoded4)
print(f"Decoded: {decoded4!r}")
print(f"Round-trip successful: {np.array_equal(decoded4, original4)}")