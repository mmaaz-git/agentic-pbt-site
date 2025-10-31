import scipy.sparse as sp
import scipy.sparse.linalg as sla
import numpy as np

print("Testing the basic reproduction case:")
A = sp.csc_array([[2.0]])
Ainv = sla.inv(A)

print(f"Input type: {type(A)}")
print(f"Output type: {type(Ainv)}")
print(f"Output value: {Ainv}")
print()

print("Testing if inv(inv(A)) fails for 1x1:")
try:
    Ainvinv = sla.inv(Ainv)
    print(f"inv(inv(A)) succeeded: {Ainvinv}")
except Exception as e:
    print(f"inv(inv(A)) failed with error: {type(e).__name__}: {e}")
print()

print("Testing for larger matrix (2x2):")
B = sp.csc_array([[2.0, 0], [0, 3.0]])
Binv = sla.inv(B)
print(f"2x2 matrix inverse type: {type(Binv)}")
print(f"2x2 matrix inverse shape: {Binv.shape}")