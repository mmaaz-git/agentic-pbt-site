import numpy as np
import numpy.char

print("Testing numpy.char.array with various whitespace characters:")
print("=" * 60)

test_cases = ['\r', '\n', '\t', '\x00', 'hello\r', 'world\n', '  spaces  ', 'tab\ttab', 'null\x00char']

for test_string in test_cases:
    arr = numpy.char.array([test_string])
    print(f"Input:  {repr(test_string)}")
    print(f"Output: {repr(str(arr[0]))}")
    print(f"Match:  {str(arr[0]) == test_string}")
    print()

print("\nTesting numpy.char.asarray with the same test cases:")
print("=" * 60)

for test_string in test_cases:
    arr = numpy.char.asarray([test_string])
    print(f"Input:  {repr(test_string)}")
    print(f"Output: {repr(str(arr[0]))}")
    print(f"Match:  {str(arr[0]) == test_string}")
    print()