import numpy as np
import numpy.strings as ns

# Test the bug case
arr = np.array(['a'])
result = ns.replace(arr, 'a', 'aa')

print("Input array:", arr)
print("Operation: replace 'a' with 'aa'")
print("Result:", result)
print("Expected:", ['aa'])
print("Match expected?", result[0] == 'aa')
print()

# Show that Python's native replace works correctly
print("Python's native str.replace:")
python_result = 'a'.replace('a', 'aa')
print(f"'a'.replace('a', 'aa') = '{python_result}'")
print()

# Test similar cases that work
print("Testing related cases:")
print("Case 1: 'ab' replace 'a' -> 'aa'")
arr2 = np.array(['ab'])
result2 = ns.replace(arr2, 'a', 'aa')
print(f"  Input: {arr2}, Result: {result2}, Expected: ['aab']")

print("Case 2: 'a' replace 'a' -> 'b' (same length)")
arr3 = np.array(['a'])
result3 = ns.replace(arr3, 'a', 'b')
print(f"  Input: {arr3}, Result: {result3}, Expected: ['b']")