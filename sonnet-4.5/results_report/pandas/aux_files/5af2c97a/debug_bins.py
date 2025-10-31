import pandas as pd
import numpy as np

values = [2.2250738585e-313, -1.0]
print(f"Input values: {values}")
print(f"Value 1 is denormal: {values[0] < 2.2250738585072014e-308}")
print(f"Min: {np.min(values)}, Max: {np.max(values)}")

# Simulate what pandas does internally
mn, mx = np.min(values), np.max(values)
print(f"Range: {mx - mn}")

# Try to create bins like pandas does
bins = np.linspace(mn, mx, 3, endpoint=True)  # 3 for 2 bins
print(f"Bins: {bins}")

# Simulate the adjustment pandas makes
adj = (mx - mn) * 0.001
print(f"Adjustment: {adj}")
bins_adj = bins.copy()
bins_adj[0] -= adj
print(f"Adjusted bins: {bins_adj}")

# Check for NaNs or issues
print(f"Any NaN in bins: {np.any(np.isnan(bins_adj))}")

# Let's see what happens when we divide by the range
print(f"\nDivision test:")
print(f"1 / (mx - mn) = {1 / (mx - mn)}")

# Test actual pd.cut to see where it fails
print("\nCalling pd.cut:")
try:
    result = pd.cut(values, bins=2)
    print(f"Result: {result}")
except ValueError as e:
    print(f"Error: {e}")