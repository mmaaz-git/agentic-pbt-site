import numpy as np

# Test around the boundary between 1e-15 and 1e-20
test_values = [1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20]

print("Finding exact boundary for numpy.random.geometric bug:")
print("=" * 60)

np.random.seed(42)
for p in test_values:
    result = np.random.geometric(p, size=1)[0]
    is_bug = (result == -9223372036854775808)
    print(f"p = {p:>8.0e}: result = {result:>20}, bug = {is_bug}")