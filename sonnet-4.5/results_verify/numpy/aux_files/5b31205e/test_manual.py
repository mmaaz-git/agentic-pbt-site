import numpy as np

p = 1e-100
np.random.seed(42)
result = np.random.geometric(p, size=1)[0]

print(f"p = {p}")
print(f"result = {result}")
print(f"Expected: positive integer >= 1")
print(f"Bug confirmed: {result == -9223372036854775808}")

# Test with different small probabilities
print("\nTesting with different small probability values:")
test_values = [1e-10, 1e-20, 1e-30, 1e-40, 1e-50, 1e-100]
for p_test in test_values:
    np.random.seed(42)
    result = np.random.geometric(p_test, size=1)[0]
    print(f"p = {p_test:e}, result = {result}")