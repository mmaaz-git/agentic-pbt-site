#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages')

import numpy as np
from scipy.io.arff._arffread import NumericAttribute

attr = NumericAttribute("test")

# Test with various array sizes
test_cases = [
    np.array([]),  # Empty array
    np.array([5.0]),  # Single element
    np.array([5.0, 10.0]),  # Two elements
    np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # Multiple elements
]

for data in test_cases:
    print(f"\nTesting with data: {data}")
    print(f"Array size: {len(data)}")

    try:
        min_val, max_val, mean_val, std_val = attr._basic_stats(data)
        print(f"  min: {min_val}")
        print(f"  max: {max_val}")
        print(f"  mean: {mean_val}")
        print(f"  std: {std_val}")

        # Compare with numpy calculations
        if len(data) > 1:
            # Manual calculation of sample std
            n = len(data)
            correction_factor = np.sqrt(n / (n - 1))
            expected_sample_std = np.std(data) * correction_factor
            print(f"  Expected sample std: {expected_sample_std}")
            print(f"  Match: {np.isclose(std_val, expected_sample_std)}")

    except ZeroDivisionError as e:
        print(f"  ZeroDivisionError: {e}")
    except Exception as e:
        print(f"  Other error: {type(e).__name__}: {e}")

# Check if the formula is correct for sample standard deviation
print("\n" + "="*50)
print("Verifying the formula for sample standard deviation:")
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
n = len(data)

# Population standard deviation
pop_std = np.std(data)
print(f"Population std (numpy): {pop_std}")

# Sample standard deviation (ddof=1)
sample_std = np.std(data, ddof=1)
print(f"Sample std (numpy with ddof=1): {sample_std}")

# Using the method's formula
nbfac = n / (n - 1)
method_std = pop_std * nbfac
print(f"Method's calculation: {pop_std} * {nbfac} = {method_std}")

# Correct formula should use sqrt(n/(n-1))
correct_factor = np.sqrt(n / (n - 1))
correct_std = pop_std * correct_factor
print(f"Correct calculation: {pop_std} * sqrt({n}/{n-1}) = {correct_std}")

print(f"\nMethod's result matches numpy sample std? {np.isclose(method_std, sample_std)}")
print(f"Correct formula matches numpy sample std? {np.isclose(correct_std, sample_std)}")