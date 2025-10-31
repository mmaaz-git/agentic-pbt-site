import numpy as np

print("Test 1: Basic example with trailing null")
print("=" * 50)
s1 = 'world\x00'
arr1 = np.array([s1], dtype=str)
print(f"Input: {repr(s1)}, length: {len(s1)}")
print(f"Stored: {repr(arr1[0])}, length: {len(arr1[0])}")
print(f"Are they equal? {s1 == arr1[0]}")
print()

print("Test 2: Multiple test cases showing the pattern")
print("=" * 50)
test_cases = [
    ('\x00', ''),           # Only null -> empty string
    ('a\x00', 'a'),         # Trailing null removed
    ('\x00b', '\x00b'),     # Leading null preserved
    ('a\x00b', 'a\x00b'),   # Middle null preserved
    ('world\x00', 'world'), # Trailing null removed
    ('test\x00\x00', 'test'), # Multiple trailing nulls
    ('\x00\x00test', '\x00\x00test'), # Multiple leading nulls
    ('te\x00\x00st', 'te\x00\x00st'), # Multiple middle nulls
]

for input_str, expected_output in test_cases:
    arr = np.array([input_str], dtype=str)
    actual = arr[0]
    matches = actual == expected_output
    print(f"{repr(input_str):20} -> {repr(actual):20} (expected: {repr(expected_output):20}) {'✓' if matches else '✗'}")

print()
print("Test 3: Workaround with object dtype")
print("=" * 50)
test_string = 'world\x00'
arr_str = np.array([test_string], dtype=str)
arr_obj = np.array([test_string], dtype=object)
print(f"Original: {repr(test_string)}")
print(f"With dtype=str: {repr(arr_str[0])} (preserves? {arr_str[0] == test_string})")
print(f"With dtype=object: {repr(arr_obj[0])} (preserves? {arr_obj[0] == test_string})")