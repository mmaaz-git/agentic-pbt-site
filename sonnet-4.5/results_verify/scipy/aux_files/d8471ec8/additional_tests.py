#!/usr/bin/env python3
import numpy as np
from scipy.integrate import cumulative_simpson, cumulative_trapezoid

def test_case(y, description):
    print(f"\n{description}")
    print(f"y = {y}")

    # Test cumulative_simpson
    simpson_result = cumulative_simpson(y, initial=0)
    simpson_diffs = np.diff(simpson_result)

    print(f"cumulative_simpson(y, initial=0) = {simpson_result}")
    print(f"Differences: {simpson_diffs}")
    print(f"Has negative difference: {np.any(simpson_diffs < 0)}")

    # Compare with cumulative_trapezoid for reference
    trapz_result = cumulative_trapezoid(y, initial=0)
    trapz_diffs = np.diff(trapz_result)

    print(f"cumulative_trapezoid(y, initial=0) = {trapz_result}")
    print(f"Differences: {trapz_diffs}")
    print(f"Has negative difference: {np.any(trapz_diffs < 0)}")

# Test original failing case
test_case(np.array([0.0, 0.0, 1.0]), "Original failing case")

# Test other simple cases
test_case(np.array([1.0, 1.0, 1.0]), "All ones (constant function)")
test_case(np.array([0.0, 1.0, 0.0]), "Peak in middle")
test_case(np.array([1.0, 0.0, 0.0]), "Decreasing from 1 to 0")
test_case(np.array([0.0, 0.0, 0.0, 1.0]), "Three zeros then one")
test_case(np.array([0.0, 0.5, 1.0]), "Linear increasing")
test_case(np.array([1.0, 0.5, 0.0]), "Linear decreasing")

# Test mathematical property: cumulative integral of non-negative should be non-decreasing
print("\n" + "=" * 60)
print("Mathematical check: For f(x) >= 0, cumulative integral should be non-decreasing")
print("=" * 60)

# Simple constant positive function
y = np.ones(10)
result = cumulative_simpson(y, initial=0)
diffs = np.diff(result)
print(f"\nConstant function y=1 over 10 points:")
print(f"All differences >= 0? {np.all(diffs >= -1e-10)}")
print(f"Min difference: {np.min(diffs)}")

# Quadratic function (which Simpson should handle exactly)
x = np.linspace(0, 1, 11)
y = x**2  # Non-negative quadratic
result = cumulative_simpson(y, initial=0)
diffs = np.diff(result)
print(f"\nQuadratic y=x^2 from 0 to 1:")
print(f"All differences >= 0? {np.all(diffs >= -1e-10)}")
print(f"Min difference: {np.min(diffs)}")