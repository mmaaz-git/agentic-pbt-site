import numpy as np
from numpy.matrixlib import bmat, matrix

# Test variations based on documentation

# Test 1: Using gdict and ldict together (should work)
X = matrix([[1, 2]])
Y = matrix([[3, 4]])

try:
    result = bmat('X', ldict={}, gdict={'X': X})
    print("Test 1 (gdict + empty ldict): SUCCESS")
    print(f"Result: {result}")
except Exception as e:
    print(f"Test 1 FAILED: {e}")

# Test 2: Using gdict alone (the bug case)
try:
    result = bmat('X', gdict={'X': X})
    print("Test 2 (gdict alone): SUCCESS")
    print(f"Result: {result}")
except Exception as e:
    print(f"Test 2 FAILED: {e}")

# Test 3: Using neither (should use frame locals/globals)
try:
    result = bmat('X')
    print("Test 3 (no dicts, use frame): SUCCESS")
    print(f"Result: {result}")
except Exception as e:
    print(f"Test 3 FAILED: {e}")

# Test 4: More complex string with gdict
A = matrix([[1, 1], [1, 1]])
B = matrix([[2, 2], [2, 2]])
C = matrix([[3, 4], [5, 6]])
D = matrix([[7, 8], [9, 0]])

try:
    result = bmat('A,B; C,D', gdict={'A': A, 'B': B, 'C': C, 'D': D})
    print("Test 4 (complex with gdict): SUCCESS")
    print(f"Result shape: {result.shape}")
except Exception as e:
    print(f"Test 4 FAILED: {e}")

# Test 5: With ldict provided as well
try:
    result = bmat('A,B; C,D', ldict={}, gdict={'A': A, 'B': B, 'C': C, 'D': D})
    print("Test 5 (complex with gdict + ldict): SUCCESS")
    print(f"Result shape: {result.shape}")
except Exception as e:
    print(f"Test 5 FAILED: {e}")