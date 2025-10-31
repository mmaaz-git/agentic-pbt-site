#!/usr/bin/env python3
"""Test script to reproduce the reported bug."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

print("Testing scipy.sparse.linalg.expm_multiply with extremely small time interval")
print("="*70)

# Set up the test case
np.random.seed(0)
n = 3
A = sp.random(n, n, density=0.3, format='csr', dtype=float) * 0.5
v = np.random.randn(n)

print(f"Matrix A shape: {A.shape}")
print(f"Vector v shape: {v.shape}")
print(f"Testing with stop=5e-324 (smallest positive float)")

try:
    # This should trigger the ZeroDivisionError according to the bug report
    result = sla.expm_multiply(A, v, start=0, stop=5e-324, num=2, endpoint=True)
    print(f"Result shape: {result.shape}")
    print(f"Result: {result}")
    print("\nNo error occurred - bug may be fixed or not reproducible")
except ZeroDivisionError as e:
    print(f"\nZeroDivisionError occurred as expected:")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"\nUnexpected error occurred:")
    print(f"  Error type: {type(e).__name__}")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("Testing with slightly larger value (1e-300)")
try:
    result2 = sla.expm_multiply(A, v, start=0, stop=1e-300, num=2, endpoint=True)
    print(f"Result shape: {result2.shape}")
    print(f"Result: {result2}")
except Exception as e:
    print(f"Error with 1e-300: {e}")

print("\n" + "="*70)
print("Testing with normal value (0.1)")
try:
    result3 = sla.expm_multiply(A, v, start=0, stop=0.1, num=2, endpoint=True)
    print(f"Result shape: {result3.shape}")
    print(f"Result: {result3}")
except Exception as e:
    print(f"Error with 0.1: {e}")

# Test the hypothesis test scenario
print("\n" + "="*70)
print("Testing semigroup property with t1=0.0, t2=5e-324")
t1 = 0.0
t2 = 5e-324

try:
    result1 = sla.expm_multiply(A, v, start=0, stop=t1 + t2, num=2, endpoint=True)[-1]
    intermediate = sla.expm_multiply(A, v, start=0, stop=t1, num=2, endpoint=True)[-1]
    result2 = sla.expm_multiply(A, intermediate, start=0, stop=t2, num=2, endpoint=True)[-1]

    relative_error = np.linalg.norm(result1 - result2) / (np.linalg.norm(result1) + 1e-10)
    print(f"Relative error: {relative_error}")
    print(f"Test passed: {relative_error < 1e-4}")
except Exception as e:
    print(f"Error in semigroup test: {e}")
    import traceback
    traceback.print_exc()