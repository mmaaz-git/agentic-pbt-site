import numpy as np
import pandas as pd

def _round_frac(x, precision: int):
    """
    Round the fractional part of the given number
    """
    if not np.isfinite(x) or x == 0:
        return x
    else:
        frac, whole = np.modf(x)
        if whole == 0:
            digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
        else:
            digits = precision
        return np.around(x, digits)

# Test with the problematic small float
small_value = 2.2250738585072014e-308
print(f"Testing _round_frac with value: {small_value}")

for precision in range(1, 10):
    print(f"\nPrecision {precision}:")
    try:
        frac, whole = np.modf(small_value)
        print(f"  frac={frac}, whole={whole}")

        if whole == 0:
            digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
            print(f"  Calculated digits: {digits}")

        result = _round_frac(small_value, precision)
        print(f"  Result: {result}")
        print(f"  Is NaN? {np.isnan(result)}")
    except Exception as e:
        print(f"  Error: {e}")

# Test np.around directly with high precision
print("\n" + "="*50)
print("Testing np.around directly with high precision values:")
for digits in [10, 50, 100, 200, 300, 310]:
    result = np.around(small_value, digits)
    print(f"np.around({small_value}, {digits}) = {result}, is NaN: {np.isnan(result)}")

# Test what happens in the actual qcut scenario
print("\n" + "="*50)
print("Testing actual bins that would be created:")
values = [0.0, 1.0, 2.0, 2.2250738585072014e-308]
series = pd.Series(values)
sorted_values = series.sort_values()
print(f"Sorted values: {sorted_values.values}")

# The quantile bins that would be created
quantiles = [0, 0.25, 0.5, 0.75, 1.0]
bins = []
for q in quantiles:
    idx = int(q * (len(sorted_values) - 1))
    bins.append(sorted_values.iloc[idx])
print(f"Bins before rounding: {bins}")

# Simulate what _format_labels does
for precision in [3, 4, 5]:  # Default precision is 3
    print(f"\nWith precision {precision}:")
    rounded_bins = [_round_frac(b, precision) for b in bins]
    print(f"  Rounded bins: {rounded_bins}")
    print(f"  Contains NaN? {any(np.isnan(b) for b in rounded_bins)}")