import numpy as np
import numpy.strings as nps

print("=" * 60)
print("Testing numpy.strings.replace with null characters")
print("=" * 60)

# Test 1: Replace null char in 'abc' (no null chars present)
arr = np.array(['abc'])
result = nps.replace(arr, '\x00', 'X')[0]
print(f"Test 1: 'abc' with replace('\\x00', 'X')")
print(f"  Expected (Python): 'abc' (no null chars to replace)")
print(f"  Actual (NumPy):    {repr(result)}")
print(f"  Match? {result == 'abc'}")
print()

# Test 2: Replace null char in 'a\x00b' (contains null char)
arr2 = np.array(['a\x00b'])
result2 = nps.replace(arr2, '\x00', 'X')[0]
print(f"Test 2: 'a\\x00b' with replace('\\x00', 'X')")
print(f"  Expected (Python): 'aXb'")
print(f"  Actual (NumPy):    {repr(result2)}")
print(f"  Match? {result2 == 'aXb'}")
print()

# Test 3: More comprehensive tests
test_cases = [
    ('hello', '\x00', 'X'),
    ('h\x00ello', '\x00', 'X'),
    ('\x00hello', '\x00', 'X'),
    ('hello\x00', '\x00', 'X'),
    ('\x00\x00', '\x00', 'X'),
]

print("Additional test cases:")
for test_str, old, new in test_cases:
    arr = np.array([test_str])
    np_result = nps.replace(arr, old, new)[0]
    py_result = test_str.replace(old, new)
    match = np_result == py_result
    print(f"  Input: {repr(test_str)}")
    print(f"    Python result: {repr(py_result)}")
    print(f"    NumPy result:  {repr(np_result)}")
    print(f"    Match? {match}")
    print()