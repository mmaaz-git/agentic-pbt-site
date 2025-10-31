import numpy as np
from pandas.core.util.hashing import hash_array

arr_obj = np.array(['a', 'b', 'c'], dtype=object)

print("Testing UTF-8 encoding:")
result_utf8 = hash_array(arr_obj, encoding='utf8', hash_key='0123456789123456')
print(f"UTF-8: {result_utf8}")

print("\nTesting UTF-16 encoding:")
try:
    result_utf16 = hash_array(arr_obj, encoding='utf16', hash_key='0123456789123456')
    print(f"UTF-16: {result_utf16}")
except ValueError as e:
    print(f"Error with UTF-16: {e}")

# Let's also verify the byte lengths
print("\nByte length verification:")
print(f"'0123456789123456' encoded as UTF-8: {len('0123456789123456'.encode('utf8'))} bytes")
print(f"'0123456789123456' encoded as UTF-16: {len('0123456789123456'.encode('utf16'))} bytes")