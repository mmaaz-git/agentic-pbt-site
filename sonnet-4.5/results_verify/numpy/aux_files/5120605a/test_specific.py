import numpy as np

# Test the specific example from the bug report
arr = np.array([1.46875144e-290] * 29)

mean_val = np.mean(arr)
min_val = np.min(arr)
max_val = np.max(arr)

print(f"Min:  {min_val:.20e}")
print(f"Mean: {mean_val:.20e}")
print(f"Max:  {max_val:.20e}")
print(f"min <= mean <= max: {min_val <= mean_val <= max_val}")
print()
print(f"Mean < Min: {mean_val < min_val}")
print(f"Mean > Max: {mean_val > max_val}")
print()

# Also test the fact that all elements are identical
print(f"All elements equal: {np.all(arr == arr[0])}")
print(f"First element: {arr[0]:.20e}")
print(f"All elements have same value: {len(set(arr.tolist())) == 1}")
print()

# Test using sum and division
manual_mean = np.sum(arr) / len(arr)
print(f"Manual mean (sum/len): {manual_mean:.20e}")
print(f"Manual mean equals np.mean: {manual_mean == mean_val}")