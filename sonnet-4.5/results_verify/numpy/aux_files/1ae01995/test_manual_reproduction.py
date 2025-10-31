import numpy as np

print("Testing null character handling in NumPy arrays:")
print("-" * 50)

s1 = '\x00'
arr1 = np.array(s1)
print(f"Input:  {repr(s1)} (length={len(s1)})")
print(f"Output: {repr(str(arr1))} (length={len(str(arr1))})")
print()

s2 = 'abc\x00'
arr2 = np.array(s2)
print(f"Input:  {repr(s2)} (length={len(s2)})")
print(f"Output: {repr(str(arr2))} (length={len(str(arr2))})")
print()

s3 = '\x00abc'
arr3 = np.array(s3)
print(f"Input:  {repr(s3)} (length={len(s3)})")
print(f"Output: {repr(str(arr3))} (length={len(str(arr3))})")
print()

# Additional test cases
s4 = 'a\x00b'
arr4 = np.array(s4)
print(f"Input:  {repr(s4)} (length={len(s4)})")
print(f"Output: {repr(str(arr4))} (length={len(str(arr4))})")
print()

# Test with numpy.char functions
import numpy.char as char
print("Testing with numpy.char.upper:")
print("-" * 50)

for test_str in ['\x00', 'abc\x00', '\x00abc', 'a\x00b']:
    upper_result = char.upper(test_str)
    print(f"Input:  {repr(test_str)} (length={len(test_str)})")
    print(f"Output: {repr(str(upper_result))} (length={len(str(upper_result))})")
    print()

# Test Python's normal string behavior for comparison
print("Python's normal string behavior (for comparison):")
print("-" * 50)
for test_str in ['\x00', 'abc\x00', '\x00abc', 'a\x00b']:
    upper_python = test_str.upper()
    print(f"Input:  {repr(test_str)} (length={len(test_str)})")
    print(f"Output: {repr(upper_python)} (length={len(upper_python)})")
    print()