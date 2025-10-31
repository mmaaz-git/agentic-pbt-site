import pandas as pd
import numpy as np

# Test the specific failing case
series = [0.0, 494699.5, 0.0, 0.0, 0.00390625]
window = 3

print(f"Testing specific case: series={series}, window={window}")

s = pd.Series(series)
rolling_var = s.rolling(window=window).var()

print(f"\nRolling variance results:")
for i, var_val in enumerate(rolling_var):
    print(f"Index {i}: {var_val}")

valid_mask = ~rolling_var.isna()
if valid_mask.sum() > 0:
    min_var = rolling_var[valid_mask].min()
    print(f"\nMinimum variance: {min_var}")
    print(f"Is minimum variance negative? {min_var < 0}")

    if min_var < -1e-10:
        print(f"\nERROR: Variance should be non-negative, got {min_var}")
    else:
        print(f"\nTest passed with tolerance of -1e-10")