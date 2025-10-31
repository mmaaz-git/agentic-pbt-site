import numpy as np

rng = np.random.Generator(np.random.PCG64(42))
result = rng.wald(1.0, 1e-10, size=100)

print(f"Negative values: {np.sum(result < 0)}/100")
print(f"Min value: {np.min(result):.6e}")
print(f"Max value: {np.max(result):.6e}")

# Show some of the negative values
negative_values = result[result < 0]
print(f"\nFirst 10 negative values:")
for i, val in enumerate(negative_values[:10]):
    print(f"  {val:.10e}")