import numpy as np

# Create an array of 29 identical values
value = 1.46875144e-290
arr = np.array([value] * 29)

# Verify all elements are identical
assert np.all(arr == arr[0]), "Not all array elements are identical"

# Calculate mean, min, and max
mean_val = np.mean(arr)
min_val = np.min(arr)
max_val = np.max(arr)

# Print results
print(f"Array length: {len(arr)}")
print(f"All elements identical: {np.all(arr == arr[0])}")
print(f"First element: {arr[0]:.20e}")
print()
print(f"Min:  {min_val:.20e}")
print(f"Mean: {mean_val:.20e}")
print(f"Max:  {max_val:.20e}")
print()
print(f"mean - max: {mean_val - max_val:.20e}")
print(f"min <= mean <= max: {min_val <= mean_val <= max_val}")

# Demonstrate the underlying issue
sum_val = np.sum(arr)
manual_mean = sum_val / len(arr)
print()
print(f"Sum of array: {sum_val:.20e}")
print(f"Manual mean (sum/len): {manual_mean:.20e}")
print(f"np.mean() result:      {mean_val:.20e}")
print(f"Manual mean == np.mean: {manual_mean == mean_val}")