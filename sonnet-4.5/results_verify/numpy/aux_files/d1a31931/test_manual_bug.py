import numpy as np
import numpy.char

test_string = '\x00'
sep = '0'

arr = np.array([test_string])

python_result = test_string.partition(sep)
numpy_result = numpy.char.partition(arr, sep)

print(f"Python partition: {python_result}")
print(f"NumPy partition:  {tuple(numpy_result[0])}")

python_rpart = test_string.rpartition(sep)
numpy_rpart = numpy.char.rpartition(arr, sep)

print(f"\nPython rpartition: {python_rpart}")
print(f"NumPy rpartition:  {tuple(numpy_rpart[0])}")

# Let's also test with a null byte in different positions
print("\n--- Testing null bytes in different positions ---")

test_cases = [
    ('\x00abc', 'b'),
    ('a\x00bc', 'b'),
    ('abc\x00', 'b'),
    ('a\x00b\x00c', 'b'),
]

for test_str, separator in test_cases:
    arr = np.array([test_str])

    python_part = test_str.partition(separator)
    numpy_part = numpy.char.partition(arr, separator)

    print(f"\nString: {repr(test_str)}, Sep: {repr(separator)}")
    print(f"  Python partition: {python_part}")
    print(f"  NumPy partition:  {tuple(numpy_part[0])}")
    print(f"  Match: {tuple(numpy_part[0]) == python_part}")