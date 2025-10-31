import numpy as np
import numpy.strings as nps

print("Testing various null character positions:")
print("-" * 60)

test_cases = [
    # (first_string, second_string, description)
    ('\x00', 'test', 'Leading null'),
    ('\x00\x00', 'abc', 'Multiple leading nulls'),
    ('a\x00', 'b', 'Trailing null'),
    ('a\x00b', 'c', 'Middle null'),
    ('\x00a', 'b', 'Leading null with content'),
    ('a\x00\x00b', 'c', 'Multiple middle nulls'),
    ('', 'test', 'Empty string'),
    ('test', '', 'Empty second string'),
    ('test', '\x00', 'Null in second operand'),
    ('\x00', '\x00', 'Both nulls'),
    ('a', 'b\x00c', 'Null in second string'),
]

for s1, s2, desc in test_cases:
    arr1 = np.array([s1], dtype=str)
    result = nps.add(arr1, s2)[0]
    expected = s1 + s2
    match = result == expected
    print(f"{desc:25} | add({repr(s1):12}, {repr(s2):10})")
    print(f"{'':25} | Expected: {repr(expected):20} Got: {repr(result):20} Match: {match}")
    print()

# Test with arrays on both sides
print("\nTesting with arrays on both sides:")
print("-" * 60)
arr1 = np.array(['\x00', 'a\x00', '\x00b'], dtype=str)
arr2 = np.array(['test', 'x', 'y'], dtype=str)
result = nps.add(arr1, arr2)
for i in range(len(arr1)):
    expected = str(arr1[i]) + str(arr2[i])
    print(f"arr[{i}]: {repr(arr1[i]):10} + {repr(arr2[i]):10} = Expected: {repr(expected):15}, Got: {repr(result[i]):15}, Match: {result[i] == expected}")