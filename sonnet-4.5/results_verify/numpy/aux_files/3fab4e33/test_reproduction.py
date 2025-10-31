import numpy as np

print("Testing basic reproduction...")
m = np.matrix([[1, 2, 3], [4, 5, 6]])
print(f"Original: shape={m.shape}, ndim={m.ndim}")

result = m[:, np.newaxis, :]
print(f"After indexing: shape={result.shape}, ndim={result.ndim}")
print(f"Is matrix: {isinstance(result, np.matrix)}")

assert result.ndim == 3
assert isinstance(result, np.matrix)
print("✓ Bug confirmed: 3D matrix object created")

print("\nTesting that 3D matrices break expected operations...")
try:
    inverse = result.I
    print("Inverse computed successfully (unexpected)")
except Exception as e:
    print(f"✓ Inverse failed as expected: {e}")