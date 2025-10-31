import numpy as np
import numpy.char

test_char = 'ẖ'
arr = np.array([test_char])

numpy_result = numpy.char.swapcase(arr)[0]
python_result = test_char.swapcase()

print(f"Input: '{test_char}' (U+1E96, LATIN SMALL LETTER H WITH LINE BELOW)")
print(f"Python str.swapcase: '{python_result}' = 'H' + COMBINING MACRON BELOW (len={len(python_result)})")
print(f"NumPy char.swapcase: '{numpy_result}' = 'H' only (len={len(numpy_result)})")
print(f"Match: {str(numpy_result) == python_result}")

print("\nOther affected characters:")
for char in ['ẖ', 'ǰ', 'ß']:
    arr = np.array([char])
    print(f"  '{char}' → NumPy: {repr(str(numpy.char.upper(arr)[0]))}, Python: {repr(char.upper())}")