import numpy as np
import scipy.sparse as sp

# Test different sizes of zero matrices
sizes = [(1, 1), (2, 2), (3, 5), (10, 10)]

for m, n in sizes:
    print(f"\nTesting {m}x{n} zero DIA matrices:")

    A = sp.dia_array(np.zeros((m, n)))
    B = sp.dia_array(np.zeros((n, m)))

    print(f"  A shape: {A.shape}, offsets: {A.offsets}, data shape: {A.data.shape}")
    print(f"  B shape: {B.shape}, offsets: {B.offsets}, data shape: {B.data.shape}")

    try:
        result = A @ B
        print(f"  Success! Result shape: {result.shape}")
    except RuntimeError as e:
        print(f"  ERROR: {e}")

    # Try workaround with CSR format
    try:
        result = A.tocsr() @ B.tocsr()
        print(f"  CSR workaround works: {result.shape}")
    except Exception as e:
        print(f"  CSR workaround failed: {e}")

# Test non-zero matrices
print("\n\nTesting non-zero DIA matrices:")
A = sp.dia_array(np.ones((3, 3)))
B = sp.dia_array(np.ones((3, 3)))
try:
    result = A @ B
    print(f"  Success! Result: {result.toarray()}")
except Exception as e:
    print(f"  ERROR: {e}")