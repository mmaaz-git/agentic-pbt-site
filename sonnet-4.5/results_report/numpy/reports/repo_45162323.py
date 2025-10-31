import numpy as np

# Reproduce the bug with numpy.random.wald producing negative values
rng = np.random.Generator(np.random.PCG64(42))
result = rng.wald(1.0, 1e-10, size=100)

print(f"Negative values: {np.sum(result < 0)}/100")
print(f"Min value: {np.min(result):.6e}")
print(f"Max value: {np.max(result):.6e}")

# Show some actual negative values
negative_values = result[result < 0]
if len(negative_values) > 0:
    print(f"\nFirst few negative values:")
    for i, val in enumerate(negative_values[:5]):
        print(f"  {val:.10e}")