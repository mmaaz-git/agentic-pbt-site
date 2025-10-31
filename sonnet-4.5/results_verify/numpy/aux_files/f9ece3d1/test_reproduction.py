import numpy as np
import numpy.strings

arr = np.array([''])

endswith_result = numpy.strings.endswith(arr, '\x00')
rfind_result = numpy.strings.rfind(arr, '\x00')

print(f"NumPy endswith([''], '\\x00'): {endswith_result[0]}")
print(f"NumPy rfind([''], '\\x00'):    {rfind_result[0]}")
print()
print(f"Python ''.endswith('\\x00'): {repr(''.endswith('\x00'))}")
print(f"Python ''.rfind('\\x00'):    {repr(''.rfind('\x00'))}")