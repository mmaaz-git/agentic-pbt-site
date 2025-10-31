#!/usr/bin/env python3
"""Verify the mathematical correctness and compare behaviors"""

import numpy as np
import pandas as pd
import warnings

def manual_variance(data, ddof=0):
    """Calculate variance manually to understand the math"""
    if len(data) == 0:
        return "Empty data - variance undefined"

    n = len(data)
    if n - ddof == 0:
        return "Division by zero: n - ddof = 0"
    if n - ddof < 0:
        return f"Negative denominator: n - ddof = {n - ddof}"

    mean = sum(data) / n
    squared_diffs = [(x - mean) ** 2 for x in data]
    sum_squared_diffs = sum(squared_diffs)

    variance = sum_squared_diffs / (n - ddof)
    return variance

print("=== Mathematical verification ===")
print("\nCase 1: [1.0, 2.0] with ddof=2")
data = [1.0, 2.0]
print(f"  Manual calculation: {manual_variance(data, ddof=2)}")
print(f"  NumPy result: {np.var(data, ddof=2)}")
print(f"  Pandas result: {pd.Series(data).var(ddof=2)}")

print("\nCase 2: Empty array with ddof=0")
data = []
print(f"  Manual calculation: {manual_variance(data, ddof=0)}")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print(f"  NumPy result: {np.var(data, ddof=0)}")
    print(f"  Pandas result: {pd.Series(data).var(ddof=0)}")

print("\nCase 3: [1.0, 2.0, 3.0] with ddof=3")
data = [1.0, 2.0, 3.0]
print(f"  Manual calculation: {manual_variance(data, ddof=3)}")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print(f"  NumPy result: {np.var(data, ddof=3)}")
    print(f"  Pandas result: {pd.Series(data).var(ddof=3)}")

print("\nCase 4: [1.0, 2.0, 3.0] with ddof=4")
data = [1.0, 2.0, 3.0]
print(f"  Manual calculation: {manual_variance(data, ddof=4)}")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print(f"  NumPy result: {np.var(data, ddof=4)}")
    print(f"  Pandas result: {pd.Series(data).var(ddof=4)}")

print("\n=== Statistical interpretation ===")
print("ddof (Delta Degrees of Freedom) represents constraints on the data:")
print("- ddof=0: Population variance (uses all N data points)")
print("- ddof=1: Sample variance (Bessel's correction, N-1 denominator)")
print("- ddof >= N: Statistically meaningless (negative or zero degrees of freedom)")
print("\nWhen ddof >= N, the calculation becomes undefined or negative,")
print("which doesn't make statistical sense.")