#!/usr/bin/env python3
import numpy as np
import numpy.char
from hypothesis import given, strategies as st

# First, let's test the Hypothesis property-based test
print("=== Testing Hypothesis property-based test ===")
@given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10))
def test_upper_matches_python(strings):
    for s in strings:
        arr = np.array([s])
        numpy_result = numpy.char.upper(arr)[0]
        python_result = s.upper()
        assert numpy_result == python_result, f"Failed for string '{s}': numpy gave '{numpy_result}', python gave '{python_result}'"

# Test with the specific failing input directly
try:
    strings = ['ﬀ']
    for s in strings:
        arr = np.array([s])
        numpy_result = numpy.char.upper(arr)[0]
        python_result = s.upper()
        assert numpy_result == python_result, f"Failed for string '{s}': numpy gave '{numpy_result}', python gave '{python_result}'"
    print("Hypothesis test with 'ﬀ' passed")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")

print("\n=== Bug 1: upper() truncates ligatures ===")
arr = np.array(['ﬀ'])
result = numpy.char.upper(arr)[0]
print(f"Input: 'ﬀ' (LATIN SMALL LIGATURE FF)")
print(f"Expected: 'FF'")
print(f"Got: '{result}'")
print(f"Length of result: {len(result)}")
print(f"Python's upper(): '{arr[0].upper()}'")

print("\n=== Bug 2: swapcase() truncates expansions ===")
arr = np.array(['ß'])
result = numpy.char.swapcase(arr)[0]
print(f"Input: 'ß' (German sharp S)")
print(f"Expected: 'SS'")
print(f"Got: '{result}'")
print(f"Length of result: {len(result)}")
print(f"Python's swapcase(): '{arr[0].swapcase()}'")

print("\n=== Bug 3: replace() truncates when replacement expands string ===")
arr = np.array(['0'])
result = numpy.char.replace(arr, '0', '00')[0]
print(f"Input: '0'")
print(f"Replace '0' with '00'")
print(f"Expected: '00'")
print(f"Got: '{result}'")
print(f"Length of result: {len(result)}")
print(f"Python's replace(): '{arr[0].replace('0', '00')}'")

print("\n=== Bug 4: translate() truncates when translation expands characters ===")
translation_table = str.maketrans({'a': 'AA'})
arr = np.array(['a'])
result = numpy.char.translate(arr, translation_table)[0]
print(f"Input: 'a'")
print(f"Translate 'a' to 'AA'")
print(f"Expected: 'AA'")
print(f"Got: '{result}'")
print(f"Length of result: {len(result)}")
print(f"Python's translate(): '{arr[0].translate(translation_table)}'")

# Let's also check the dtype information
print("\n=== Additional dtype investigation ===")
arr = np.array(['ﬀ'])
print(f"Original array dtype: {arr.dtype}")
result_arr = numpy.char.upper(arr)
print(f"Result array dtype: {result_arr.dtype}")

arr2 = np.array(['hello'])
print(f"'hello' array dtype: {arr2.dtype}")
result_arr2 = numpy.char.upper(arr2)
print(f"'HELLO' result array dtype: {result_arr2.dtype}")