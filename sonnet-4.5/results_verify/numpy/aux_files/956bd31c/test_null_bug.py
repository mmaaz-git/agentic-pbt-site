import numpy as np

print("Testing null character truncation in NumPy arrays")
print("="*50)

# Test 1: Single null character
arr = np.array(['\x00'])
print(f"Test 1: Single null character")
print(f"  Input: {repr('\x00')}")
print(f"  Output: {repr(arr[0])}")
print(f"  Expected: {repr('\x00')}")
print(f"  Match: {arr[0] == '\x00'}")
print()

# Test 2: String with trailing null
arr = np.array(['hello\x00'])
print(f"Test 2: String with trailing null")
print(f"  Input: {repr('hello\x00')}")
print(f"  Output: {repr(arr[0])}")
print(f"  Expected: {repr('hello\x00')}")
print(f"  Match: {arr[0] == 'hello\x00'}")
print()

# Test 3: String with null in middle
arr = np.array(['hello\x00world'])
print(f"Test 3: String with null in middle")
print(f"  Input: {repr('hello\x00world')}")
print(f"  Output: {repr(arr[0])}")
print(f"  Expected: {repr('hello\x00world')}")
print(f"  Match: {arr[0] == 'hello\x00world'}")
print()

# Test 4: Multiple trailing nulls
arr = np.array(['test\x00\x00\x00'])
print(f"Test 4: Multiple trailing nulls")
print(f"  Input: {repr('test\x00\x00\x00')}")
print(f"  Output: {repr(arr[0])}")
print(f"  Expected: {repr('test\x00\x00\x00')}")
print(f"  Match: {arr[0] == 'test\x00\x00\x00'}")