import numpy as np

p = 1e-100
np.random.seed(42)
result = np.random.geometric(p, size=10)

print(f"Testing numpy.random.geometric with p = {p}")
print(f"Generated 10 values: {result}")
print(f"Minimum value: {np.min(result)}")
print(f"Maximum value: {np.max(result)}")
print(f"All values >= 1 (expected): {np.all(result >= 1)}")
print(f"")
print(f"INT64_MIN = -9223372036854775808")
print(f"Bug confirmed (values equal INT64_MIN): {np.all(result == -9223372036854775808)}")