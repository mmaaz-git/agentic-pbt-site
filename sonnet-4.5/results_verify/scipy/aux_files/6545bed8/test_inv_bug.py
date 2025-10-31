import numpy as np
from scipy import sparse
from scipy.sparse import linalg

print("Testing scipy.sparse.linalg.inv return types")
print("=" * 50)

# Test with 1x1 matrix
A1 = sparse.diags([2.0], offsets=0, format='csr')
print(f"A1 (1x1 matrix):\n{A1.toarray()}")
inv1 = linalg.inv(A1)
print(f"inv1 type: {type(inv1)}")
print(f"inv1 is sparse: {sparse.issparse(inv1)}")
if not sparse.issparse(inv1):
    print(f"inv1 value (dense): {inv1}")
else:
    print(f"inv1 value (sparse): {inv1.toarray()}")

print("\n" + "-" * 50 + "\n")

# Test with 2x2 matrix
A2 = sparse.diags([2.0, 3.0], offsets=0, format='csr')
print(f"A2 (2x2 matrix):\n{A2.toarray()}")
inv2 = linalg.inv(A2)
print(f"inv2 type: {type(inv2)}")
print(f"inv2 is sparse: {sparse.issparse(inv2)}")
if not sparse.issparse(inv2):
    print(f"inv2 value (dense): {inv2}")
else:
    print(f"inv2 value (sparse):\n{inv2.toarray()}")

print("\n" + "-" * 50 + "\n")

# Test with 3x3 matrix
A3 = sparse.diags([1.0, 2.0, 3.0], offsets=0, format='csr')
print(f"A3 (3x3 matrix):\n{A3.toarray()}")
inv3 = linalg.inv(A3)
print(f"inv3 type: {type(inv3)}")
print(f"inv3 is sparse: {sparse.issparse(inv3)}")
if not sparse.issparse(inv3):
    print(f"inv3 value (dense): {inv3}")
else:
    print(f"inv3 value (sparse):\n{inv3.toarray()}")

print("\n" + "=" * 50 + "\n")
print("SUMMARY:")
print(f"1x1 matrix returns sparse: {sparse.issparse(inv1)}")
print(f"2x2 matrix returns sparse: {sparse.issparse(inv2)}")
print(f"3x3 matrix returns sparse: {sparse.issparse(inv3)}")