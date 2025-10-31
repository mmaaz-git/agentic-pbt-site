import numpy as np
import numpy.strings as nps

# Test the specific failing case mentioned in the bug report
print("Testing specific failing case: ['\\x00']")
strings = ['\x00']
arr = np.array(strings, dtype=str)
scalar = 'test'
result = nps.add(arr, scalar)
expected = strings[0] + scalar

print(f"Input: {repr(strings[0])}")
print(f"Scalar: {repr(scalar)}")
print(f"Expected: {repr(expected)}")
print(f"Got: {repr(result[0])}")
print(f"Match: {result[0] == expected}")
print()

# Now run the full reproducing code from the bug report
print("Running full reproducing code from bug report:")
print("-" * 50)

test_cases = [
    ('\x00', 'test'),
    ('\x00\x00', 'abc'),
    ('a\x00', 'b'),
]

for s1, s2 in test_cases:
    arr1 = np.array([s1], dtype=str)
    result = nps.add(arr1, s2)[0]
    expected = s1 + s2
    print(f"add({repr(s1):10}, {repr(s2):8}): Expected={repr(expected):15}, Got={repr(result):15}, Match={result == expected}")