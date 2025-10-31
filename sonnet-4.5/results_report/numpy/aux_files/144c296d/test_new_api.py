import numpy as np

# Test the new Generator API with the same problematic alpha values
rng = np.random.default_rng(42)
alpha = [0.001953125, 0.00390625]
samples = rng.dirichlet(alpha, size=1000)

print(f"Testing new Generator API with alpha={alpha}")
print(f"Total samples generated: {len(samples)}")
print(f"Samples with NaN: {np.isnan(samples).any(axis=1).sum()}")
print(f"Samples with inf: {np.isinf(samples).any(axis=1).sum()}")

sums = samples.sum(axis=1)
print(f"All sums close to 1.0: {np.allclose(sums, 1.0)}")
print(f"Min sum: {sums.min():.10f}")
print(f"Max sum: {sums.max():.10f}")