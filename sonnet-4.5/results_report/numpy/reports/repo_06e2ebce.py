import numpy as np
import numpy.strings

# Test case: empty string with null character
arr = np.array([''])

# NumPy results
endswith_result = numpy.strings.endswith(arr, '\x00')
rfind_result = numpy.strings.rfind(arr, '\x00')

print(f"NumPy endswith([''], '\\x00'): {endswith_result[0]}")
print(f"NumPy rfind([''], '\\x00'):    {rfind_result[0]}")
print()

# Python results for comparison
print(f"Python ''.endswith('\\x00'): {repr(''.endswith('\x00'))}")
print(f"Python ''.rfind('\\x00'):    {repr(''.rfind('\x00'))}")
print()

# Additional tests to understand the behavior
print("Additional test cases:")
print(f"NumPy endswith(['a'], '\\x00'): {numpy.strings.endswith(np.array(['a']), '\x00')[0]}")
print(f"NumPy rfind(['a'], '\\x00'):    {numpy.strings.rfind(np.array(['a']), '\x00')[0]}")
print(f"Python 'a'.endswith('\\x00'): {repr('a'.endswith('\x00'))}")
print(f"Python 'a'.rfind('\\x00'):    {repr('a'.rfind('\x00'))}")