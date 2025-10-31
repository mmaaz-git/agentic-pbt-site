import numpy.char as char
import numpy as np

# Get the docstring for char.replace
print("NumPy char.replace docstring:")
print("="*60)
print(char.replace.__doc__)
print("="*60)

# Check what numpy.char.replace is supposed to do
# Let's also check if it's element-wise as claimed
arr = np.array(['0', 'hello', 'test'])
result = char.replace(arr, '00', 'REPLACEMENT')
print("\nElement-wise test on array:")
print(f"Input array: {arr}")
print(f"char.replace(arr, '00', 'REPLACEMENT'): {result}")

# Compare with Python's behavior element-wise
py_results = [s.replace('00', 'REPLACEMENT') for s in arr]
print(f"Python's replace element-wise: {py_results}")