#!/usr/bin/env python3
import numpy as np
import numpy.strings as nps

print("=== Testing numpy.strings.replace with empty string pattern ===")
print()

# Test 1: Basic reproduction
print("Test 1: Basic reproduction case")
arr = np.array([''])
result = nps.replace(arr, '', '00')
python_result = ''.replace('', '00')
print(f"Python: {''.replace('', '00')!r}")
print(f"NumPy:  {result[0]!r}")
print(f"Match: {result[0] == python_result}")
print()

# Test 2: Comprehensive test cases
print("Test 2: Comprehensive demonstration of the bug pattern:")
test_cases = [
    ('', 'ab'),
    ('', 'abc'),
    ('a', 'xyz'),
    ('ab', 'xyz'),
    ('abc', 'XYZ'),
    ('hello', 'XYZ'),
]

for input_str, replacement in test_cases:
    python_result = input_str.replace('', replacement)
    numpy_result = nps.replace(np.array([input_str]), '', replacement)[0]
    match = "✓" if python_result == numpy_result else "✗"
    print(f"{match} input={input_str!r:10} | Python={python_result!r:30} | NumPy={numpy_result!r}")

print()
print("=== Testing Python's behavior with empty string replacement ===")
print()

# Test Python's str.replace with empty strings
test_strings = ['', 'a', 'ab', 'abc', 'abcd', 'hello']
replacement = 'X'

for s in test_strings:
    result = s.replace('', replacement)
    print(f"'{s}'.replace('', '{replacement}') = '{result}'")
    print(f"  Length: input={len(s)}, output={len(result)}, expected_insertions={len(s)+1}")

print()
print("=== Testing with different replacement lengths ===")
print()

for repl_len in [1, 2, 3, 5]:
    replacement = 'X' * repl_len
    for input_str in ['', 'a', 'ab']:
        python_result = input_str.replace('', replacement)
        numpy_result = nps.replace(np.array([input_str]), '', replacement)[0]
        match = "✓" if python_result == numpy_result else "✗"
        print(f"{match} input='{input_str}' repl='{replacement}' | Python={len(python_result)} chars | NumPy={len(numpy_result)} chars")

print()
print("=== Running Hypothesis test ===")
print()

try:
    from hypothesis import given, strategies as st

    @given(st.text(max_size=2), st.text(min_size=2, max_size=10))
    def test_replace_empty_pattern_short_strings(input_str, replacement):
        arr = np.array([input_str])
        numpy_result = nps.replace(arr, '', replacement)[0]
        python_result = input_str.replace('', replacement)
        assert numpy_result == python_result, f"Mismatch: input={input_str!r}, repl={replacement!r}, numpy={numpy_result!r}, python={python_result!r}"

    test_replace_empty_pattern_short_strings()
    print("Hypothesis test completed without finding issues")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")
except Exception as e:
    print(f"Hypothesis test error: {e}")