import numpy as np

print("Testing NumPy's diag with large k values:")

# Test numpy.diag with various offsets
m1 = np.eye(3)
print(f"Original 3x3 identity matrix:\n{m1}")

# Extract diagonal at k=3
diag1 = np.diag(m1, k=3)
print(f"\nnp.diag(eye(3), k=3): {diag1}")
print(f"Shape: {diag1.shape}")

# Extract diagonal at k=4
diag2 = np.diag(m1, k=4)
print(f"\nnp.diag(eye(3), k=4): {diag2}")
print(f"Shape: {diag2.shape}")

# Test creating diagonal with large k
print("\n\nCreating diagonal arrays:")
d1 = np.diag([], k=10)
print(f"np.diag([], k=10):\n{d1}")
print(f"Shape: {d1.shape}")

# Test with actual values
d2 = np.diag([1, 2, 3], k=0)
print(f"\nnp.diag([1,2,3], k=0):\n{d2}")

d3 = np.diag([1, 2], k=3)
print(f"\nnp.diag([1,2], k=3):\n{d3}")