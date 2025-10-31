import numpy as np
from scipy.spatial import distance

x = np.array([False, False, False])
y = np.array([False, False, False])

result = distance.dice(x, y)

print(f"dice({x}, {y}) = {result}")
print(f"Expected: 0.0 (identical arrays should have distance 0)")
print(f"Actual: {result}")
print(f"Is NaN? {np.isnan(result)}")