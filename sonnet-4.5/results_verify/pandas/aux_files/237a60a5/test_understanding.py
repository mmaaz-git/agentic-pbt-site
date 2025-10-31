#!/usr/bin/env python3
"""Test to understand how qcut handles duplicates='drop'"""

import pandas as pd
import numpy as np

# Test case from bug report
data = [0.0, 0.0, 1.0, 2.0, 3.0]
print("Test data:", data)

# First, let's see what quantiles are calculated
quantiles = np.linspace(0, 1, 5)  # For q=4, we get 5 quantile points
print("\nQuantiles for q=4:", quantiles)

# Calculate the actual quantile values
series = pd.Series(data)
quantile_values = series.quantile(quantiles)
print("Quantile values:", quantile_values.tolist())
print("Unique quantile values:", quantile_values.unique().tolist())

# This explains why we get 3 bins instead of 4
print("\nBecause 0.0 appears twice as quantile values (at 0% and 25%),")
print("when duplicates='drop' is used, these get merged into a single bin.")

# Now test with qcut
result = pd.qcut(data, q=4, duplicates='drop', retbins=True)
print("\nqcut result with retbins=True:")
print("Categories:", result[0].categories.tolist())
print("Bin edges:", result[1])
print("Value counts:", result[0].value_counts().tolist())

# Let's also check what happens with duplicates='raise'
print("\n" + "="*60)
print("Testing with duplicates='raise' (should fail):")
try:
    result_raise = pd.qcut(data, q=4, duplicates='raise')
    print("Unexpected: No error raised!")
except ValueError as e:
    print(f"ValueError raised as expected: {str(e)[:100]}...")