import numpy.char as char
import numpy as np

# Test with scalar strings
print("Testing with scalar strings:")
s = '0'
old = '00'
new = 'REPLACEMENT'

# Direct scalar
result_scalar = char.replace(s, old, new)
print(f"char.replace('{s}', '{old}', '{new}') = {repr(result_scalar)}")
print(f"Type: {type(result_scalar)}")
print(f"Result as string: {repr(str(result_scalar))}")

# Using numpy array with single element
arr = np.array(['0'])
result_array = char.replace(arr, old, new)
print(f"\nchar.replace(np.array(['{s}']), '{old}', '{new}') = {repr(result_array)}")
print(f"First element: {repr(result_array[0])}")

# Check numpy version
print(f"\nNumPy version: {np.__version__}")