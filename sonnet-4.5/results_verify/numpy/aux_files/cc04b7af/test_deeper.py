import numpy as np
import numpy.rec as rec

# Test different string patterns containing null
test_cases = [
    '\x00',           # Just null
    'a\x00b',         # Null in middle
    '\x00abc',        # Null at start
    'abc\x00',        # Null at end
    'ab\x00\x00cd',   # Multiple nulls
]

print("Testing string truncation with numpy.rec.fromrecords:")
for test_str in test_cases:
    records = [(1, test_str, 2.0)]
    result = rec.fromrecords(records, names=['a', 'b', 'c'])
    print(f"  Input: {repr(test_str):20} -> Output: {repr(str(result[0].b)):20} (lost {len(test_str) - len(str(result[0].b))} chars)")

print("\nTesting direct numpy array creation:")
for test_str in test_cases:
    arr = np.array([test_str])
    print(f"  Input: {repr(test_str):20} -> Output: {repr(str(arr[0])):20} (lost {len(test_str) - len(str(arr[0]))} chars)")

print("\nTesting with explicit Unicode dtype:")
for test_str in test_cases:
    arr = np.array([test_str], dtype='U10')
    print(f"  Input: {repr(test_str):20} -> Output: {repr(str(arr[0])):20} (lost {len(test_str) - len(str(arr[0]))} chars)")

print("\nTesting with object dtype:")
for test_str in test_cases:
    arr = np.array([test_str], dtype=object)
    print(f"  Input: {repr(test_str):20} -> Output: {repr(arr[0]):20} (preserves? {test_str == arr[0]})")