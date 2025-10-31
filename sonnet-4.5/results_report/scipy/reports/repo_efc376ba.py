import numpy as np
import scipy.spatial.distance as dist

u = np.array([False, False, False, False, False])
v = np.array([False, False, False, False, False])

result = dist.dice(u, v)

assert np.array_equal(u, v), f"Arrays are not equal: u={u}, v={v}"
print(f"dice(u, u) = {result}")
print(f"Is result NaN? {np.isnan(result)}")
print(f"Arrays are identical: {np.array_equal(u, v)}")