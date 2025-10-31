import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages')

import numpy as np
from scipy.spatial import geometric_slerp

start = np.array([1.0, 0.0, 0.0])
end_same = np.array([1.0, 0.0, 0.0])
end_different = np.array([0.0, 1.0, 0.0])

result_same = geometric_slerp(start, end_same, 0.5)
result_different = geometric_slerp(start, end_different, 0.5)

print(f"When start == end: shape = {result_same.shape}")
print(f"When start != end: shape = {result_different.shape}")
print(f"Expected: Both should have shape (3,)")

# Check actual values
print(f"\nWhen start == end: result = {result_same}")
print(f"When start != end: result = {result_different}")

# Test with different scalar t values
print("\nTesting with different scalar t values:")
for t_val in [0.0, 0.3, 0.5, 0.7, 1.0]:
    res1 = geometric_slerp(start, end_same, t_val)
    res2 = geometric_slerp(start, end_different, t_val)
    print(f"t={t_val}: same endpoints shape={res1.shape}, different endpoints shape={res2.shape}")