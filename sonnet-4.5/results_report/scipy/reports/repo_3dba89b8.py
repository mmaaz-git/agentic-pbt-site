from scipy.integrate import cumulative_simpson
import numpy as np

y = np.array([0.0, 0.0, 1.0])

cumulative_result = cumulative_simpson(y, initial=0)
diffs = np.diff(cumulative_result)

print(f"y = {y}")
print(f"cumulative_simpson(y, initial=0) = {cumulative_result}")
print(f"Differences between consecutive values: {diffs}")
print(f"Has negative difference: {np.any(diffs < 0)}")