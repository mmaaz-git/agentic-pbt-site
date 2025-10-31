import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env')

import numpy as np
import numpy.char as char

# Test 1: Example from the bug report
print("=== Test 1: Example from bug report ===")
arr = np.array(['hello\x00'])
result = char.multiply(arr, 2)[0]
expected = 'hello\x00' * 2

print(f"char.multiply result: {repr(result)}")
print(f"Expected: {repr(expected)}")
print(f"Match: {result == expected}")

# Test 2: Minimal case from property test
print("\n=== Test 2: Minimal case from property test ===")
arr = np.array(['0\x00'])
result = char.multiply(arr, 1)[0]
expected = '0\x00' * 1

print(f"char.multiply result: {repr(result)}")
print(f"Expected: {repr(expected)}")
print(f"Match: {result == expected}")

# Test 3: Multiple null bytes
print("\n=== Test 3: Multiple null bytes ===")
arr = np.array(['abc\x00\x00\x00'])
result = char.multiply(arr, 3)[0]
expected = 'abc\x00\x00\x00' * 3

print(f"char.multiply result: {repr(result)}")
print(f"Expected: {repr(expected)}")
print(f"Match: {result == expected}")

# Test 4: Check Python's native behavior for comparison
print("\n=== Test 4: Python native string multiplication ===")
s = 'hello\x00'
py_result = s * 2
print(f"Python * result: {repr(py_result)}")
print(f"Contains null bytes: {'\\x00' in py_result}")