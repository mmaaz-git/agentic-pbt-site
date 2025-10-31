import numpy as np
import numpy.strings

# Test various null character behaviors
print("Testing null character behavior in NumPy strings:")
print("=" * 60)

# Test 1: Direct array creation with null characters
arr1 = np.array(['\x00', 'a\x00', '\x00b', 'a\x00b'])
print(f"Array with nulls: {repr(arr1)}")
print(f"Dtype: {arr1.dtype}")
print(f"Element lengths: {[len(s) for s in arr1]}")

# Test 2: Adding null characters
arr2 = np.array(['a', 'b', 'c'])
result = numpy.strings.add(arr2, '\x00')
print(f"\nAdding null to ['a', 'b', 'c']: {repr(result)}")
print(f"Lengths after adding null: {[len(s) for s in result]}")

# Test 3: Null at different positions
arr3 = np.array([''])
tests = [
    ('start', '\x00abc'),
    ('middle', 'a\x00bc'),
    ('end', 'abc\x00'),
    ('only null', '\x00'),
    ('multiple nulls', '\x00\x00'),
]

print("\nAdding strings with nulls at different positions to ['']:")
for desc, test_str in tests:
    result = numpy.strings.add(arr3, test_str)
    print(f"  {desc:15} '{repr(test_str)}' -> {repr(result[0])}, len={len(result[0])}")

# Test 4: Check if this is consistent with Python string behavior
print("\nPython string concatenation comparison:")
s1 = ''
s2 = '\x00'
s3 = '0'
py_result1 = (s1 + s2) + s3
py_result2 = s1 + (s2 + s3)
print(f"Python: ('' + '\\x00') + '0' = {repr(py_result1)}")
print(f"Python: '' + ('\\x00' + '0') = {repr(py_result2)}")
print(f"Python preserves associativity: {py_result1 == py_result2}")