import numpy as np
import numpy.char

print("Testing np.char.find with null bytes:")
print("="*50)

# Test 1: Empty string searching for null byte
arr = np.array([''])
result = np.char.find(arr, '\x00')
print(f"np.char.find([''], '\\x00') = {result[0]}")
print(f"Expected (Python): ''.find('\\x00') = {('').find('\x00')}")
print(f"Match: {result[0] == ('').find('\x00')}")
print()

# Test 2: String with null byte in middle
arr2 = np.array(['a\x00b'])
result2 = np.char.find(arr2, '\x00')
print(f"np.char.find(['a\\x00b'], '\\x00') = {result2[0]}")
print(f"Expected (Python): 'a\\x00b'.find('\\x00') = {('a\x00b').find('\x00')}")
print(f"Match: {result2[0] == ('a\x00b').find('\x00')}")
print()

# Test 3: Count null bytes in regular string
arr3 = np.array(['hello'])
result3 = np.char.count(arr3, '\x00')
print(f"np.char.count(['hello'], '\\x00') = {result3[0]}")
print(f"Expected (Python): 'hello'.count('\\x00') = {('hello').count('\x00')}")
print(f"Match: {result3[0] == ('hello').count('\x00')}")
print()

# Test 4: Startswith null byte
arr4 = np.array(['hello'])
result4 = np.char.startswith(arr4, '\x00')
print(f"np.char.startswith(['hello'], '\\x00') = {result4[0]}")
print(f"Expected (Python): 'hello'.startswith('\\x00') = {('hello').startswith('\x00')}")
print(f"Match: {result4[0] == ('hello').startswith('\x00')}")
print()

# Additional tests
print("Additional tests:")
print("-"*30)

# Test 5: rfind with null byte
arr5 = np.array(['hello'])
result5 = np.char.rfind(arr5, '\x00')
print(f"np.char.rfind(['hello'], '\\x00') = {result5[0]}")
print(f"Expected (Python): 'hello'.rfind('\\x00') = {('hello').rfind('\x00')}")
print(f"Match: {result5[0] == ('hello').rfind('\x00')}")
print()

# Test 6: endswith with null byte
arr6 = np.array(['hello'])
result6 = np.char.endswith(arr6, '\x00')
print(f"np.char.endswith(['hello'], '\\x00') = {result6[0]}")
print(f"Expected (Python): 'hello'.endswith('\\x00') = {('hello').endswith('\x00')}")
print(f"Match: {result6[0] == ('hello').endswith('\x00')}")
print()

# Test 7: index with null byte (should raise ValueError)
print("Testing np.char.index with null byte:")
arr7 = np.array([''])
try:
    result7 = np.char.index(arr7, '\x00')
    print(f"np.char.index([''], '\\x00') = {result7[0]} (should have raised ValueError)")
except ValueError as e:
    print(f"np.char.index([''], '\\x00') raised ValueError as expected: {e}")

try:
    python_result = ''.index('\x00')
except ValueError as e:
    print(f"Python ''.index('\\x00') raised ValueError as expected: {e}")
print()

# Test with regular substrings (non-null) to show normal behavior works
print("Control test with normal substring:")
print("-"*30)
arr8 = np.array(['hello world'])
result8 = np.char.find(arr8, 'world')
print(f"np.char.find(['hello world'], 'world') = {result8[0]}")
print(f"Expected (Python): 'hello world'.find('world') = {('hello world').find('world')}")
print(f"Match: {result8[0] == ('hello world').find('world')}")