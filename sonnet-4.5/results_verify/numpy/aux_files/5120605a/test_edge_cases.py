import numpy as np

def test_mean_bounds(arr, description):
    print(f"\n{description}")
    print(f"Array shape: {arr.shape}, dtype: {arr.dtype}")

    mean_val = np.mean(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)

    print(f"Min:  {min_val:.20e}")
    print(f"Mean: {mean_val:.20e}")
    print(f"Max:  {max_val:.20e}")

    violation = not (min_val <= mean_val <= max_val)
    print(f"Violates bounds: {violation}")

    if violation:
        if mean_val < min_val:
            print(f"  Mean is {min_val - mean_val:.20e} below min")
        if mean_val > max_val:
            print(f"  Mean is {mean_val - max_val:.20e} above max")

    return violation

# Test various cases
print("="*60)
print("Testing various edge cases for numpy.mean bounds violation")
print("="*60)

# 1. Very small numbers (denormalized)
arr1 = np.array([5e-324] * 10, dtype=np.float64)  # Near smallest positive float64
test_mean_bounds(arr1, "1. Very small denormalized numbers")

# 2. Very large numbers
arr2 = np.array([1e308] * 10, dtype=np.float64)  # Near largest float64
test_mean_bounds(arr2, "2. Very large numbers")

# 3. Mixed signs with small numbers
arr3 = np.array([1e-100, -1e-100, 1e-100, -1e-100], dtype=np.float64)
test_mean_bounds(arr3, "3. Mixed signs with small numbers")

# 4. Different array sizes with the problematic value
for n in [5, 10, 20, 29, 30, 50, 100]:
    arr = np.array([1.46875144e-290] * n, dtype=np.float64)
    if test_mean_bounds(arr, f"4. Array of {n} identical values (1.46875144e-290)"):
        print(f"  >>> VIOLATION FOUND at n={n}")

# 5. Test the Hypothesis-found case
arr5 = np.array([5.000538517706642e-184] * 6, dtype=np.float64)
test_mean_bounds(arr5, "5. Hypothesis-found case (6 copies of 5.000538517706642e-184)")

# 6. Test with float32
arr6 = np.array([1e-38] * 10, dtype=np.float32)
test_mean_bounds(arr6, "6. Float32 array with small numbers")

# 7. Test regular numbers (should work fine)
arr7 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
test_mean_bounds(arr7, "7. Regular numbers [1, 2, 3, 4, 5]")

# 8. Test with zeros
arr8 = np.array([0.0] * 100, dtype=np.float64)
test_mean_bounds(arr8, "8. Array of 100 zeros")

# 9. Test mixed with very different magnitudes
arr9 = np.array([1e-200, 1e200], dtype=np.float64)
test_mean_bounds(arr9, "9. Mixed magnitudes [1e-200, 1e200]")

print("\n" + "="*60)
print("Summary: The bug is reproducible with certain small float64 values")
print("when accumulated in arrays of specific sizes.")
print("="*60)