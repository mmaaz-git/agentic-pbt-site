import numpy as np
import numpy.strings as ns

# Test case 1: slice with explicit None stop and step
arr = np.array(["hello", "world"])
result1 = ns.slice(arr, 0, None, 2)
print("Test 1: ns.slice(['hello', 'world'], 0, None, 2)")
print("  Result:", result1)
print("  Expected: ['hlo', 'wrd'] (from 'hello'[0:None:2] = 'hlo')")

# Test case 2: minimal failing example from Hypothesis
arr2 = np.array(['0'])
result2 = ns.slice(arr2, 0, None, 2)
print("\nTest 2: ns.slice(['0'], 0, None, 2)")
print("  Result:", result2)
print("  Expected: ['0'] (from '0'[0:None:2] = '0')")

# Test case 3: slice with explicit None stop, no step
arr3 = np.array(["hello", "world"])
result3 = ns.slice(arr3, 2, None)
print("\nTest 3: ns.slice(['hello', 'world'], 2, None)")
print("  Result:", result3)
print("  Expected: ['llo', 'rld'] (from 'hello'[2:None] = 'llo')")

# Demonstrating correct Python behavior for comparison
print("\n--- Python's built-in slicing behavior ---")
print("'hello'[0:None:2] =", 'hello'[0:None:2])
print("'world'[0:None:2] =", 'world'[0:None:2])
print("'0'[0:None:2] =", '0'[0:None:2])
print("'hello'[2:None] =", 'hello'[2:None])
print("'world'[2:None] =", 'world'[2:None])