import numpy as np

test_values = [1e-5, 1e-10, 1e-15, 1e-20, 1e-25, 1e-30, 1e-40, 1e-50, 1e-100]

print("Testing threshold for numpy.random.geometric bug:")
print("=" * 60)

np.random.seed(42)
for p in test_values:
    result = np.random.geometric(p, size=1)[0]
    is_bug = (result == -9223372036854775808)
    print(f"p = {p:>8.0e}: result = {result:>20}, bug = {is_bug}")