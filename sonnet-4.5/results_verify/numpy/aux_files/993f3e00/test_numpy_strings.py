import numpy as np

# Test how numpy handles null characters in strings
print("=== Testing NumPy String Handling with Null Characters ===\n")

# Test 1: Leading null character
test1 = '\x00'
arr1 = np.array([test1], dtype='U100')
print(f"Test 1 - Leading null:")
print(f"  Input Python string: {test1!r}, len={len(test1)}")
print(f"  NumPy array value: {arr1[0]!r}, len={len(arr1[0])}")
print(f"  Are they equal? {arr1[0] == test1}")

# Test 2: Trailing null character
test2 = 'abc\x00'
arr2 = np.array([test2], dtype='U100')
print(f"\nTest 2 - Trailing null:")
print(f"  Input Python string: {test2!r}, len={len(test2)}")
print(f"  NumPy array value: {arr2[0]!r}, len={len(arr2[0])}")
print(f"  Are they equal? {arr2[0] == test2}")

# Test 3: Embedded null character
test3 = 'abc\x00def'
arr3 = np.array([test3], dtype='U100')
print(f"\nTest 3 - Embedded null:")
print(f"  Input Python string: {test3!r}, len={len(test3)}")
print(f"  NumPy array value: {arr3[0]!r}, len={len(arr3[0])}")
print(f"  Are they equal? {arr3[0] == test3}")

# Test 4: Multiple null characters
test4 = '\x00\x00abc\x00'
arr4 = np.array([test4], dtype='U100')
print(f"\nTest 4 - Multiple nulls:")
print(f"  Input Python string: {test4!r}, len={len(test4)}")
print(f"  NumPy array value: {arr4[0]!r}, len={len(arr4[0])}")
print(f"  Are they equal? {arr4[0] == test4}")

# Test 5: Only null characters
test5 = '\x00\x00\x00'
arr5 = np.array([test5], dtype='U100')
print(f"\nTest 5 - Only nulls:")
print(f"  Input Python string: {test5!r}, len={len(test5)}")
print(f"  NumPy array value: {arr5[0]!r}, len={len(arr5[0])}")
print(f"  Are they equal? {arr5[0] == test5}")

# Direct test of np.str_ behavior
print("\n=== Direct np.str_ Construction ===")
s1 = np.str_('\x00')
s2 = np.str_('abc\x00')
s3 = np.str_('abc\x00def')
print(f"np.str_('\\x00'): {s1!r}, len={len(s1)}")
print(f"np.str_('abc\\x00'): {s2!r}, len={len(s2)}")
print(f"np.str_('abc\\x00def'): {s3!r}, len={len(s3)}")