import numpy as np
import numpy.strings as nps

# Test various null character positions
test_cases = [
    '\x00',           # Leading null
    '\x00abc',        # Leading null with text
    'abc\x00',        # Trailing null
    'abc\x00def',     # Middle null
    '\x00\x00',       # Multiple nulls
    'a\x00b\x00c',    # Multiple nulls with text
]

print("Testing null character handling in numpy arrays:")
print("=" * 50)
for test_str in test_cases:
    arr = np.array([test_str], dtype='<U100')
    print(f"Input: {repr(test_str)}")
    print(f"  Length in Python: {len(test_str)}")
    print(f"  Array value: {repr(arr[0])}")
    print(f"  Length in NumPy: {len(arr[0])}")
    print(f"  Match: {arr[0] == test_str}")
    print()

# Test with other string operations
print("\nTesting string operations with null characters:")
print("=" * 50)

# Test multiply
arr = np.array(['\x00'], dtype='<U100')
result = nps.multiply(arr, 3)
print(f"multiply(['\x00'], 3):")
print(f"  NumPy result: {repr(result[0])}")
print(f"  Python result: {repr('\x00' * 3)}")
print(f"  Match: {result[0] == '\x00' * 3}")
print()

# Test ljust
arr = np.array(['\x00'], dtype='<U100')
result = nps.ljust(arr, 5, fillchar='X')
print(f"ljust(['\x00'], 5, 'X'):")
print(f"  NumPy result: {repr(result[0])}")
print(f"  Python result: {repr('\x00'.ljust(5, 'X'))}")
print(f"  Match: {result[0] == '\x00'.ljust(5, 'X')}")
print()

# Test rjust
arr = np.array(['\x00'], dtype='<U100')
result = nps.rjust(arr, 5, fillchar='X')
print(f"rjust(['\x00'], 5, 'X'):")
print(f"  NumPy result: {repr(result[0])}")
print(f"  Python result: {repr('\x00'.rjust(5, 'X'))}")
print(f"  Match: {result[0] == '\x00'.rjust(5, 'X')}")