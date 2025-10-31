import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages')

import numpy as np
from scipy.spatial import geometric_slerp

# Test case demonstrating the shape inconsistency
start = np.array([1.0, 0.0, 0.0])
end_same = np.array([1.0, 0.0, 0.0])  # Same as start
end_different = np.array([0.0, 1.0, 0.0])  # Different from start

# Test with scalar t = 0.5
t_scalar = 0.5

# When start == end
result_same = geometric_slerp(start, end_same, t_scalar)
print(f"When start == end:")
print(f"  Input t: {t_scalar} (type: {type(t_scalar).__name__})")
print(f"  Result shape: {result_same.shape}")
print(f"  Result value: {result_same}")
print(f"  Result type: {type(result_same).__name__}")
print()

# When start != end
result_different = geometric_slerp(start, end_different, t_scalar)
print(f"When start != end:")
print(f"  Input t: {t_scalar} (type: {type(t_scalar).__name__})")
print(f"  Result shape: {result_different.shape}")
print(f"  Result value: {result_different}")
print(f"  Result type: {type(result_different).__name__}")
print()

print(f"Shape mismatch detected: {result_same.shape} vs {result_different.shape}")
print(f"Expected: Both should have shape (3,) for scalar t")