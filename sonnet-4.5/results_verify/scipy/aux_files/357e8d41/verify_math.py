import numpy as np

print("Mathematical verification of standard deviation for single element:")
print("=" * 60)

single_element = np.array([5.0])

print(f"\nArray: {single_element}")
print(f"Size: {single_element.size}")

# Population standard deviation (ddof=0)
print(f"\nPopulation standard deviation (ddof=0):")
pop_std = np.std(single_element, ddof=0)
print(f"  np.std(data, ddof=0) = {pop_std}")
print(f"  Formula: sqrt(sum((x - mean)^2) / N)")
print(f"  For single element: sqrt((5.0 - 5.0)^2 / 1) = sqrt(0 / 1) = 0")

# Sample standard deviation (ddof=1)
print(f"\nSample standard deviation (ddof=1):")
try:
    sample_std = np.std(single_element, ddof=1)
    print(f"  np.std(data, ddof=1) = {sample_std}")
except:
    print(f"  np.std(data, ddof=1) raises an error or returns NaN")
    # Manual calculation
    sample_std_manual = np.sqrt(np.sum((single_element - np.mean(single_element))**2) / (len(single_element) - 1))
    print(f"  Manual calculation: sqrt(0 / 0) = {sample_std_manual}")

# What _basic_stats is trying to do
print(f"\nWhat _basic_stats does:")
print(f"  nbfac = n / (n - 1) = 1 / 0 = undefined (division by zero)")
print(f"  std = np.std(data) * nbfac = 0 * inf = ?")

# Check what other libraries return
print(f"\nWhat different approaches yield for single element:")
print(f"  Population std (mathematically correct): 0.0")
print(f"  Sample std (undefined, n-1=0): NaN or undefined")
print(f"  Current scipy behavior: ZeroDivisionError")

# Check sample std with numpy directly
print(f"\nDirect numpy calculation with ddof=1:")
sample_std_numpy = np.std([5.0], ddof=1)
print(f"  np.std([5.0], ddof=1) = {sample_std_numpy}")
print(f"  Is NaN? {np.isnan(sample_std_numpy)}")
print(f"  Is finite? {np.isfinite(sample_std_numpy)}")