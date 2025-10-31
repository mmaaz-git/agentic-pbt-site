#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env')

import numpy as np
import numpy.char as char

# Test case 1: Basic example from the bug report
print("=== Test Case 1: Basic Example ===")
arr = np.array(['hello\x00'])
result = char.multiply(arr, 2)[0]
expected = 'hello\x00' * 2

print(f"Input: 'hello\\x00' * 2")
print(f"char.multiply result: {repr(result)}")
print(f"Expected: {repr(expected)}")
print(f"Match: {result == expected}")
print(f"Result length: {len(result)}")
print(f"Expected length: {len(expected)}")
print()

# Test case 2: Minimal failing case from hypothesis
print("=== Test Case 2: Minimal Case ===")
arr = np.array(['0\x00'])
result = char.multiply(arr, 1)[0]
expected = '0\x00' * 1

print(f"Input: '0\\x00' * 1")
print(f"char.multiply result: {repr(result)}")
print(f"Expected: {repr(expected)}")
print(f"Match: {result == expected}")
print(f"Result length: {len(result)}")
print(f"Expected length: {len(expected)}")
print()

# Test case 3: Multiple null bytes
print("=== Test Case 3: Multiple Null Bytes ===")
arr = np.array(['abc\x00\x00\x00'])
result = char.multiply(arr, 3)[0]
expected = 'abc\x00\x00\x00' * 3

print(f"Input: 'abc\\x00\\x00\\x00' * 3")
print(f"char.multiply result: {repr(result)}")
print(f"Expected: {repr(expected)}")
print(f"Match: {result == expected}")
print(f"Result length: {len(result)}")
print(f"Expected length: {len(expected)}")
print()

# Test case 4: Only null bytes
print("=== Test Case 4: Only Null Bytes ===")
arr = np.array(['\x00'])
result = char.multiply(arr, 5)[0]
expected = '\x00' * 5

print(f"Input: '\\x00' * 5")
print(f"char.multiply result: {repr(result)}")
print(f"Expected: {repr(expected)}")
print(f"Match: {result == expected}")
print(f"Result length: {len(result)}")
print(f"Expected length: {len(expected)}")
print()

# Test case 5: Python native string multiplication for comparison
print("=== Python Native String Multiplication ===")
test_strings = ['hello\x00', '0\x00', 'abc\x00\x00\x00', '\x00']
for s in test_strings:
    python_result = s * 3
    print(f"'{repr(s)}' * 3 = {repr(python_result)} (length: {len(python_result)})")