import numpy as np
import numpy.strings as nps

print("Testing numpy.strings.replace bug")
print("="*50)

arr = np.array(['0'])
print(f"Input array: {arr}")
print(f"Array dtype: {arr.dtype}")
print(f"Old string: '0'")
print(f"New string: '00'")
print(f"Count: 1")
print()

result = nps.replace(arr, '0', '00', count=1)
print(f"Result: {result}")
print(f"Result dtype: {result.dtype}")
print(f"Expected: ['00']")
print(f"Bug present: {str(result[0]) != '00'}")
print()

# Test Python's string.replace for comparison
python_result = '0'.replace('0', '00', 1)
print(f"Python str.replace('0', '00', 1): '{python_result}'")
print()

# Additional test cases
print("Additional test cases:")
print("-"*30)

# Test case 2: Longer initial string
arr2 = np.array(['hello'])
result2 = nps.replace(arr2, 'l', 'LL', count=2)
expected2 = 'hello'.replace('l', 'LL', 2)
print(f"arr=['hello'], replace('l', 'LL', count=2)")
print(f"  NumPy result: {result2[0]}")
print(f"  Python result: {expected2}")
print(f"  Match: {str(result2[0]) == expected2}")
print()

# Test case 3: Empty replacement
arr3 = np.array(['test'])
result3 = nps.replace(arr3, 't', '', count=1)
expected3 = 'test'.replace('t', '', 1)
print(f"arr=['test'], replace('t', '', count=1)")
print(f"  NumPy result: '{result3[0]}'")
print(f"  Python result: '{expected3}'")
print(f"  Match: {str(result3[0]) == expected3}")
print()

# Test case 4: Multiple elements
arr4 = np.array(['a', 'aa', 'aaa'])
result4 = nps.replace(arr4, 'a', 'bb', count=1)
print(f"arr=['a', 'aa', 'aaa'], replace('a', 'bb', count=1)")
for i in range(len(arr4)):
    expected = arr4[i].replace('a', 'bb', 1)
    print(f"  '{arr4[i]}' -> NumPy: '{result4[i]}', Python: '{expected}', Match: {str(result4[i]) == expected}")