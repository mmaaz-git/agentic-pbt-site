import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings

# First, let's reproduce the specific failing example
print("=== Testing specific failing example ===")
arr = np.array(['0'], dtype='<U1')
start = 0
result = nps.slice(arr, start, None)
print(f"Input: arr={arr}, start={start}")
print(f"Result of nps.slice(arr, 0, None): {result}")
print(f"Expected (like Python s[0:]): ['0']")
print()

# Let's test the basic example from the bug report
print("=== Testing basic example ===")
arr = np.array(['hello'], dtype='U')
result = nps.slice(arr, 0, None)
print(f"Result of nps.slice(arr, 0, None): {result}")

s = 'hello'
print(f"Result of Python s[0:None]: {s[0:None]}")
print(f"Result of Python s[0:]: {s[0:]}")
print()

# Let's test various combinations
print("=== Testing various combinations ===")
test_str = np.array(['abcdef'], dtype='U')

print(f"Original string: {test_str}")
print(f"nps.slice(test_str, 2): {nps.slice(test_str, 2)}")  # Should be [:2] = 'ab'
print(f"nps.slice(test_str, 2, None): {nps.slice(test_str, 2, None)}")  # Should be [2:] = 'cdef'
print(f"nps.slice(test_str, 2, 4): {nps.slice(test_str, 2, 4)}")  # Should be [2:4] = 'cd'
print()

# Test with Python's slice object for comparison
print("=== Python slice behavior ===")
s = "abcdef"
print(f"s[:2] = {s[:2]}")
print(f"s[2:] = {s[2:]}")
print(f"s[2:None] = {s[2:None]}")
print(f"s[2:4] = {s[2:4]}")
print()

# Let's also test what happens when we explicitly pass None vs not passing
print("=== Testing parameter passing ===")
print("When calling with just start=2:")
print(f"  nps.slice(test_str, 2) = {nps.slice(test_str, 2)}")
print("When calling with start=2, stop=None:")
print(f"  nps.slice(test_str, 2, None) = {nps.slice(test_str, 2, None)}")