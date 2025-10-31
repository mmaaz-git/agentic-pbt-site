import numpy as np
from scipy.spatial.distance import jensenshannon

# Test case that crashes: all-zero vectors
p = np.array([0.0, 0.0, 0.0])
q = np.array([0.0, 0.0, 0.0])

print("Testing jensenshannon with all-zero vectors:")
print(f"p = {p}")
print(f"q = {q}")

result = jensenshannon(p, q)
print(f"jensenshannon(p, q) = {result}")
print(f"Result is NaN: {np.isnan(result)}")

# For comparison, test with valid probability vectors
print("\n" + "="*50)
print("Testing jensenshannon with valid probability vectors:")
p_valid = np.array([0.3, 0.3, 0.4])
q_valid = np.array([0.3, 0.3, 0.4])
print(f"p_valid = {p_valid}")
print(f"q_valid = {q_valid}")

result_valid = jensenshannon(p_valid, q_valid)
print(f"jensenshannon(p_valid, q_valid) = {result_valid}")
print(f"Result is approximately 0: {np.isclose(result_valid, 0.0)}")