import numpy as np
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

# Let's understand how numpy handles null bytes with different dtypes
test_str = '\x00'
test_bytes = test_str.encode('utf-8')

print(f"Original string: {test_str!r} (len={len(test_str)})")
print(f"Encoded bytes: {test_bytes!r} (len={len(test_bytes)})")
print()

# Create array with dtype=bytes
arr_bytes = np.array([test_bytes], dtype=bytes)
print(f"np.array([test_bytes], dtype=bytes): {arr_bytes!r}")
print(f"  dtype: {arr_bytes.dtype}")
print(f"  item: {arr_bytes[0]!r}")
print()

# Create array with dtype=object
arr_obj = np.array([test_bytes], dtype=object)
print(f"np.array([test_bytes], dtype=object): {arr_obj!r}")
print(f"  dtype: {arr_obj.dtype}")
print(f"  item: {arr_obj[0]!r}")
print()

# Test with multiple strings
test_strs = ['\x00', 'a\x00b', '\x00hello', 'hello\x00']
test_bytes_list = [s.encode('utf-8') for s in test_strs]
print("Testing multiple strings:")
for i, (s, b) in enumerate(zip(test_strs, test_bytes_list)):
    arr_b = np.array([b], dtype=bytes)
    print(f"{i}: '{s!r}' -> {b!r} -> np.array dtype=bytes: {arr_b!r} (dtype={arr_b.dtype})")