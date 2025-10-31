#!/usr/bin/env python3
"""Test to see if zsqrt can receive scalar inputs in real pandas usage"""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

# Create some test data
print("Testing real usage scenarios that might produce scalar inputs to zsqrt")
print("=" * 70)

# Test case 1: Simple EWM standard deviation
print("\nTest 1: EWM standard deviation on single value series")
s = pd.Series([5.0])  # Single value
try:
    ewm_std = s.ewm(span=2).std()
    print(f"Input series: {s.values}")
    print(f"EWM std result: {ewm_std.values}")
except Exception as e:
    print(f"Error: {e}")

# Test case 2: EWM correlation with single values
print("\nTest 2: EWM correlation with single-row data")
df = pd.DataFrame({'x': [1.0], 'y': [2.0]})
try:
    # This might produce scalar variance values internally
    corr = df['x'].ewm(span=2).corr(df['y'])
    print(f"Input DataFrame:\n{df}")
    print(f"Correlation result: {corr.values}")
except Exception as e:
    print(f"Error: {e}")

# Test case 3: Check if variance can be scalar
print("\nTest 3: Checking variance computation")
print("Looking at what x_var * y_var produces in correlation...")

# Simulate what happens inside ewm.corr
x = pd.Series([1.0, 2.0, 3.0])
y = pd.Series([2.0, 4.0, 6.0])

# When computing correlation, internally it does:
# cov / zsqrt(x_var * y_var)
# Let's see what x_var * y_var looks like

# Using exponential weighted variance
x_var = x.ewm(span=2).var()
y_var = y.ewm(span=2).var()
product = x_var * y_var

print(f"x_var type: {type(x_var)}, values: {x_var.values}")
print(f"y_var type: {type(y_var)}, values: {y_var.values}")
print(f"x_var * y_var type: {type(product)}, values: {product.values}")
print(f"Each element type: {type(product.values[0])}")

# The product is still a Series, not a scalar
# So when does it become scalar?

print("\nTest 4: Direct variance calculation (internal simulation)")
# Let's look deeper at what _cov returns
from pandas.core.window.ewm import _cov

# Create simple arrays
x_array = np.array([1.0])
y_array = np.array([2.0])
com = 1.0  # corresponds to span=2

# This simulates what happens internally
print(f"Input arrays: x={x_array}, y={y_array}")
cov_result = _cov(x_array, x_array, com)
print(f"_cov result type: {type(cov_result)}, value: {cov_result}")

# Check if it's a scalar
if np.isscalar(cov_result):
    print("_cov CAN return a scalar!")
    print("This means x_var * y_var could be scalar when passed to zsqrt")