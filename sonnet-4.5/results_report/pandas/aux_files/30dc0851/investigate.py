import pandas as pd
import numpy as np

values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.15625]

# Test with different precision values
for precision in [3, 6, 10]:
    result = pd.cut(values, bins=2, precision=precision, retbins=True)
    categories, bins = result
    print(f"\nPrecision={precision}")
    print(f"  Bins: {bins}")
    print(f"  Last interval: {categories[9]}")
    print(f"  Value 1.15625 in interval? {1.15625 in categories[9]}")

# Show what happens without precision rounding
print("\nInternal computation details:")
mn, mx = min(values), max(values)
print(f"  Min value: {mn}")
print(f"  Max value: {mx}")
print(f"  Range: {mx - mn}")
adj = (mx - mn) * 0.001  # 0.1% adjustment
print(f"  Adjustment (0.1% of range): {adj}")
print(f"  Adjusted min: {mn - adj}")
print(f"  Adjusted max: {mx}")

# Calculate actual bins before rounding
bins_no_round = np.linspace(mn - adj, mx, 3, endpoint=True)
print(f"  Bins before rounding: {bins_no_round}")
print(f"  Bins after rounding to 3 decimals: {np.round(bins_no_round, 3)}")