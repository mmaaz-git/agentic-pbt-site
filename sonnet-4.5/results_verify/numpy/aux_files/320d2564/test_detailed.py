import numpy as np
import numpy.char

print("Detailed testing of whitespace stripping behavior:")
print("=" * 70)

# Test different whitespace positions
test_cases = [
    ('  leading', 'Leading spaces'),
    ('trailing  ', 'Trailing spaces'),
    ('  both  ', 'Both leading and trailing'),
    ('mid  dle', 'Space in the middle'),
    ('\r', 'Carriage return alone'),
    ('\n', 'Newline alone'),
    ('\t', 'Tab alone'),
    ('\r\n', 'CRLF'),
    ('hello\r', 'Carriage return at end'),
    ('\rhello', 'Carriage return at beginning'),
    ('hello\n', 'Newline at end'),
    ('\nhello', 'Newline at beginning'),
    ('hello\tworld', 'Tab in middle'),
    ('\thello', 'Tab at beginning'),
    ('hello\t', 'Tab at end'),
    ('hello\r\nworld', 'CRLF in middle'),
    ('hello\x00world', 'Null in middle'),
    ('\x00', 'Null alone'),
    ('hello\x00', 'Null at end'),
    ('\x00hello', 'Null at beginning'),
    ('', 'Empty string'),
    (' ', 'Single space'),
    ('  ', 'Two spaces'),
    ('\n\n', 'Two newlines'),
    ('\r\n\r\n', 'Two CRLF'),
]

for test_string, description in test_cases:
    arr = numpy.char.array([test_string])
    output = str(arr[0])

    print(f"{description:30} | Input: {repr(test_string):20} | Output: {repr(output):20} | Equal: {output == test_string}")

print("\n\nTesting array creation vs indexing:")
print("=" * 70)

# Create array and check values before and after indexing
test_strings = ['\r', 'hello\n', '  spaces  ', '\ttest']
arr = numpy.char.array(test_strings)

print("Original input list:", test_strings)
print("Array dtype:", arr.dtype)
print("Array shape:", arr.shape)
print("\nComparing values:")
for i, orig in enumerate(test_strings):
    indexed_val = arr[i]
    str_val = str(indexed_val)
    print(f"Index {i}: Original={repr(orig):15} | arr[{i}]={repr(str_val):15} | Equal={str_val == orig}")