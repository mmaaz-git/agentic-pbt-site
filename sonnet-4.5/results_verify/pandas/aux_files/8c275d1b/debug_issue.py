import numpy as np
from pandas.io.formats.format import format_percentiles, get_precision

# Test the problematic case
percentiles = np.array([0.0, 7.506590166045388e-253])
print(f"Original percentiles: {percentiles}")
print(f"Are they different? {percentiles[0] != percentiles[1]}")

# Convert to percentage
percentiles_pct = 100 * percentiles
print(f"\nPercentiles * 100: {percentiles_pct}")

# Get precision
prec = get_precision(percentiles_pct)
print(f"Precision from get_precision: {prec}")

# Round with precision
percentiles_round_type = percentiles_pct.round(prec).astype(int)
print(f"Rounded to int: {percentiles_round_type}")

# Check if close to integers
int_idx = np.isclose(percentiles_round_type, percentiles_pct)
print(f"Are they close to integers? {int_idx}")
print(f"All close to integers? {np.all(int_idx)}")

# If we follow the early return path
if np.all(int_idx):
    out = percentiles_round_type.astype(str)
    result_early = [i + "%" for i in out]
    print(f"\nEarly return result: {result_early}")
    print(f"Are outputs unique? {len(result_early) == len(set(result_early))}")

# What would happen if we didn't have early return?
unique_pcts = np.unique(percentiles_pct)
print(f"\nUnique percentiles: {unique_pcts}")
prec2 = get_precision(unique_pcts)
print(f"Precision from unique percentiles: {prec2}")

# Now format using the actual function
result = format_percentiles([0.0, 7.506590166045388e-253])
print(f"\nActual function result: {result}")
print(f"Are outputs unique? {len(result) == len(set(result))}")