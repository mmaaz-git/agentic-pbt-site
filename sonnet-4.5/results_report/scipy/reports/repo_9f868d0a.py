from scipy.spatial.distance import dice
import numpy as np

u = np.array([False, False, False])
v = np.array([False, False, False])

result = dice(u, v)
print(f"dice([False, False, False], [False, False, False]) = {result}")

assert result == 0.0, f"Expected 0.0, got {result}"