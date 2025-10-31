import scipy.sparse as sp
import scipy.sparse.linalg as sla
import numpy as np

# Test case demonstrating the bug
A = sp.csc_array([[2.0]])
print(f"Input A type: {type(A)}")
print(f"Input A shape: {A.shape}")
print(f"Input A value: {A.toarray()}")

# First inversion
Ainv = sla.inv(A)
print(f"\nFirst inv(A) type: {type(Ainv)}")
print(f"First inv(A) shape: {Ainv.shape if hasattr(Ainv, 'shape') else 'N/A'}")
print(f"First inv(A) value: {Ainv}")

# Try second inversion - this will fail
try:
    Ainvinv = sla.inv(Ainv)
    print(f"\nSecond inv(inv(A)) type: {type(Ainvinv)}")
    print(f"Second inv(inv(A)) value: {Ainvinv}")
except TypeError as e:
    print(f"\nError on second inversion: {e}")

# Show that for 2x2 matrices it works correctly
print("\n--- For comparison, 2x2 matrix behavior ---")
A2 = sp.csc_array([[2.0, 0.0], [0.0, 3.0]])
print(f"Input A2 type: {type(A2)}")
print(f"Input A2 shape: {A2.shape}")

A2inv = sla.inv(A2)
print(f"First inv(A2) type: {type(A2inv)}")
print(f"First inv(A2) shape: {A2inv.shape}")

A2invinv = sla.inv(A2inv)
print(f"Second inv(inv(A2)) type: {type(A2invinv)}")
print(f"Second inv(inv(A2)) successful - no error")