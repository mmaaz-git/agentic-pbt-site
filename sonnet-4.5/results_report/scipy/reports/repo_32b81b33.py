#!/usr/bin/env python3
"""Minimal reproduction of scipy.sparse.eye_array inconsistent offset bounds checking bug."""

import scipy.sparse as sp

# Test case 1: k = n (should work and return empty matrix)
print("Test 1: eye_array(3, k=3, format='csr')")
try:
    E1 = sp.eye_array(3, k=3, format='csr')
    print(f"  Success: nnz={E1.nnz}, shape={E1.shape}")
except ValueError as e:
    print(f"  Error: {e}")

# Test case 2: k = n+1 (currently fails with ValueError)
print("\nTest 2: eye_array(3, k=4, format='csr')")
try:
    E2 = sp.eye_array(3, k=4, format='csr')
    print(f"  Success: nnz={E2.nnz}, shape={E2.shape}")
except ValueError as e:
    print(f"  Error: {e}")

# Test case 3: k = -n (should work and return empty matrix)
print("\nTest 3: eye_array(3, k=-3, format='csr')")
try:
    E3 = sp.eye_array(3, k=-3, format='csr')
    print(f"  Success: nnz={E3.nnz}, shape={E3.shape}")
except ValueError as e:
    print(f"  Error: {e}")

# Test case 4: k = -(n+1) (currently fails with ValueError)
print("\nTest 4: eye_array(3, k=-4, format='csr')")
try:
    E4 = sp.eye_array(3, k=-4, format='csr')
    print(f"  Success: nnz={E4.nnz}, shape={E4.shape}")
except ValueError as e:
    print(f"  Error: {e}")

# Demonstrating the minimal failing case
print("\nMinimal failing case: eye_array(1, k=2, format='csr')")
try:
    E_min = sp.eye_array(1, k=2, format='csr')
    print(f"  Success: nnz={E_min.nnz}, shape={E_min.shape}")
except ValueError as e:
    print(f"  Error: {e}")