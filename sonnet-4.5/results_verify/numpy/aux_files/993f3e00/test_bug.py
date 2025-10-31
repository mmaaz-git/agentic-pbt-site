import numpy as np
import numpy.char as char
from hypothesis import given, settings, strategies as st

# First, test the basic reproduction case
print("=== Basic Bug Reproduction ===")
arr = np.array(['\x00'], dtype='U100')
result = char.upper(arr)

print(f"Input: {arr[0]!r}, len={len(arr[0])}")
print(f"Result: {result[0]!r}, len={len(str(result[0]))}")
print(f"Python behavior: {'\x00'.upper()!r}")

# Check if this is actually a bug or expected behavior
print("\n=== Testing if null char is preserved in numpy array creation ===")
test_str = '\x00'
numpy_arr = np.array([test_str], dtype='U100')
print(f"Original string: {test_str!r}, len={len(test_str)}")
print(f"String in numpy array: {numpy_arr[0]!r}, len={len(numpy_arr[0])}")

# Test with embedded nulls
print("\n=== Testing with embedded nulls ===")
test_str2 = 'abc\x00def'
numpy_arr2 = np.array([test_str2], dtype='U100')
print(f"Original string: {test_str2!r}, len={len(test_str2)}")
print(f"String in numpy array: {numpy_arr2[0]!r}, len={len(numpy_arr2[0])}")

# Test case operations with different strings
print("\n=== Testing case operations on various strings ===")
test_cases = [
    '\x00',
    'abc\x00def',
    '\x00abc',
    'ABC',
    'abc'
]

for s in test_cases:
    arr = np.array([s], dtype='U100')
    upper_result = char.upper(arr)
    lower_result = char.lower(arr)

    print(f"\nInput: {s!r}")
    print(f"  Array value: {arr[0]!r}")
    print(f"  char.upper: {upper_result[0]!r}")
    print(f"  char.lower: {lower_result[0]!r}")
    print(f"  Python upper: {s.upper()!r}")
    print(f"  Python lower: {s.lower()!r}")

# Run the hypothesis test
print("\n=== Running Hypothesis Test ===")
@settings(max_examples=10)  # Reduced for testing
@given(st.text(alphabet='\x00abcABC', min_size=1, max_size=10))
def test_upper_preserves_null_chars(s):
    arr = np.array([s], dtype='U100')
    result = char.upper(arr)

    expected = s.upper()
    actual = str(result[0])

    # First check if numpy even preserves the null in the array
    stored_value = str(arr[0])
    if stored_value != s:
        print(f"  NumPy array doesn't preserve input: {s!r} -> {stored_value!r}")
        return  # Skip this case as it's a numpy array issue, not char.upper

    assert actual == expected, f"Input: {s!r}, Expected: {expected!r}, Got: {actual!r}"

try:
    test_upper_preserves_null_chars()
    print("Hypothesis test passed!")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")