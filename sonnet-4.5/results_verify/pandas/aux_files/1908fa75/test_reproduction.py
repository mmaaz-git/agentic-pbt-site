#!/usr/bin/env python3
"""Test to reproduce the pandas describe() percentile formatting bug"""

import pandas as pd
import numpy as np

print("Testing pandas describe() with extremely small percentile values")
print("=" * 60)

# Create test series
series = pd.Series([1, 2, 3, 4, 5])

# Test with extremely small percentile
print("\n1. Testing with percentiles=[5e-324]:")
result = series.describe(percentiles=[5e-324])
print("Result index:", result.index.tolist())
print("Full result:")
print(result)

# Check if '50%' is in the index
print(f"\nDoes '50%' appear in index? {'50%' in result.index}")

# Test with other small percentiles
print("\n2. Testing with percentiles=[1e-300]:")
result2 = series.describe(percentiles=[1e-300])
print("Result index:", result2.index.tolist())

print("\n3. Testing with percentiles=[1e-100]:")
result3 = series.describe(percentiles=[1e-100])
print("Result index:", result3.index.tolist())

print("\n4. Testing with percentiles=[1e-10]:")
result4 = series.describe(percentiles=[1e-10])
print("Result index:", result4.index.tolist())

print("\n5. Testing with normal percentiles=[0.25, 0.75]:")
result5 = series.describe(percentiles=[0.25, 0.75])
print("Result index:", result5.index.tolist())