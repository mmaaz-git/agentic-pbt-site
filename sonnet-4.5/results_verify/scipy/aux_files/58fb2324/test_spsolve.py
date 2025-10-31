import scipy.sparse as sp
import scipy.sparse.linalg as sla
import numpy as np

# Test what spsolve returns for 1x1 vs 2x2 matrices
print("Testing spsolve behavior:")

# 1x1 case
A1 = sp.csc_array([[2.0]])
I1 = sp.identity(1, format='csc')
result1 = sla.spsolve(A1, I1)
print(f"1x1 case:")
print(f"  A shape: {A1.shape}")
print(f"  I shape: {I1.shape}")
print(f"  spsolve result type: {type(result1)}")
print(f"  spsolve result shape: {result1.shape if hasattr(result1, 'shape') else 'N/A'}")
print(f"  spsolve result: {result1}")

# 2x2 case
A2 = sp.csc_array([[2.0, 0], [0, 3.0]])
I2 = sp.identity(2, format='csc')
result2 = sla.spsolve(A2, I2)
print(f"\n2x2 case:")
print(f"  A shape: {A2.shape}")
print(f"  I shape: {I2.shape}")
print(f"  spsolve result type: {type(result2)}")
print(f"  spsolve result shape: {result2.shape if hasattr(result2, 'shape') else 'N/A'}")
print(f"  Is sparse: {sp.issparse(result2)}")

# Test with a 1x1 vector RHS
print("\n1x1 with vector RHS:")
b1 = np.array([1.0])
result1b = sla.spsolve(A1, b1)
print(f"  spsolve result type: {type(result1b)}")
print(f"  spsolve result: {result1b}")