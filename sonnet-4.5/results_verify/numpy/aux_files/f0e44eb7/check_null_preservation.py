import numpy as np
import numpy.strings as nps

# Create arrays with explicit null characters
arr1 = np.array(['a'], dtype='U2')
arr2 = np.array(['a\x00'], dtype='U2')
arr3 = np.array(['a\x00b'], dtype='U3')

print("Testing null preservation in NumPy arrays:")
print(f"arr1: {repr(arr1)}, dtype={arr1.dtype}")
print(f"arr2: {repr(arr2)}, dtype={arr2.dtype}")
print(f"arr3: {repr(arr3)}, dtype={arr3.dtype}")
print()

# Check the actual memory/bytes representation
print("String representations:")
print(f"arr1[0]: {repr(arr1[0])}, str: '{str(arr1[0])}', len: {len(str(arr1[0]))}")
print(f"arr2[0]: {repr(arr2[0])}, str: '{str(arr2[0])}', len: {len(str(arr2[0]))}")
print(f"arr3[0]: {repr(arr3[0])}, str: '{str(arr3[0])}', len: {len(str(arr3[0]))}")
print()

# Check if we can encode and get the null back
print("Encoding to bytes:")
for i, arr in enumerate([arr1, arr2, arr3], 1):
    try:
        encoded = str(arr[0]).encode('utf-8')
        print(f"arr{i}[0] encoded: {encoded}")
    except Exception as e:
        print(f"arr{i}[0] encode error: {e}")
print()

# Direct comparison with Python strings
s1, s2, s3 = 'a', 'a\x00', 'a\x00b'
print("Python string comparisons:")
print(f"'{s1}' < '{s2}': {s1 < s2}")
print(f"'{s1}' < '{s3}': {s1 < s3}")
print(f"'{s2}' < '{s3}': {s2 < s3}")
print()

# NumPy strings comparisons
print("NumPy strings.less comparisons:")
print(f"arr1 < arr2: {nps.less(arr1, arr2)[0]}")
print(f"arr1 < arr3: {nps.less(arr1, arr3)[0]}")
print(f"arr2 < arr3: {nps.less(arr2, arr3)[0]}")