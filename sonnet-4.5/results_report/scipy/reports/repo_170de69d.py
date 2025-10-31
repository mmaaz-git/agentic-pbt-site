import numpy as np
import scipy.linalg

n = 19
P = scipy.linalg.pascal(n)
P_inv = scipy.linalg.invpascal(n, exact=False)

product = P @ P_inv
I = np.eye(n)

print(f"Testing n={n}")
print(f"||P @ P_inv - I||_F = {np.linalg.norm(product - I, 'fro')}")
print(f"Max absolute error: {np.max(np.abs(product - I))}")

print("\nNon-zero off-diagonal elements in P @ P_inv:")
for i in range(n):
    for j in range(n):
        if i != j and abs(product[i, j]) > 1e-8:
            print(f"  [{i}, {j}] = {product[i, j]}")

print("\nDiagonal elements that differ from 1.0:")
for i in range(n):
    if abs(product[i, i] - 1.0) > 1e-8:
        print(f"  [{i}, {i}] = {product[i, i]} (should be 1.0)")