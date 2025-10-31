import numpy as np

# Test with specific failing value from bug report
shape = 0.0009791878680225864
rng = np.random.Generator(np.random.PCG64(42))
result = rng.gamma(shape, size=100)

print(f"Testing with shape={shape}")
print(f"Zeros: {np.sum(result == 0)}/100")
print(f"Min value: {np.min(result):.6e}")
print(f"Max value: {np.max(result):.6e}")

# Also test with the example from the bug report
print("\nTesting with shape=0.001:")
rng = np.random.Generator(np.random.PCG64(42))
result = rng.gamma(0.001, size=100)

print(f"Zeros: {np.sum(result == 0)}/100")
print(f"Min value: {np.min(result):.6e}")
print(f"Max value: {np.max(result):.6e}")