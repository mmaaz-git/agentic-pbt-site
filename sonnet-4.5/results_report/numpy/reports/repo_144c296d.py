import numpy as np

np.random.seed(42)
alpha = [0.001953125, 0.00390625]
samples = np.random.dirichlet(alpha, size=1000)

print(f"Alpha values: {alpha}")
print(f"Total samples generated: {len(samples)}")
print(f"Samples with NaN: {np.isnan(samples).any(axis=1).sum()}")
print(f"Samples with inf: {np.isinf(samples).any(axis=1).sum()}")
print(f"Samples with -inf: {np.isneginf(samples).any(axis=1).sum()}")

# Show some invalid samples
invalid_mask = np.isnan(samples).any(axis=1) | np.isinf(samples).any(axis=1)
invalid_samples = samples[invalid_mask]

if len(invalid_samples) > 0:
    print(f"\nFirst 5 invalid samples:")
    for i, sample in enumerate(invalid_samples[:5]):
        print(f"  Sample {i+1}: {sample}")
        print(f"    Sum: {np.nansum(sample)}")
        print(f"    Has NaN: {np.isnan(sample).any()}, Has Inf: {np.isinf(sample).any()}")

# Check valid samples
valid_samples = samples[~invalid_mask]
if len(valid_samples) > 0:
    print(f"\nValid samples: {len(valid_samples)}")
    sums = valid_samples.sum(axis=1)
    print(f"  Min sum: {sums.min():.10f}")
    print(f"  Max sum: {sums.max():.10f}")
    print(f"  Mean sum: {sums.mean():.10f}")
    print(f"  All sums close to 1.0: {np.allclose(sums, 1.0)}")