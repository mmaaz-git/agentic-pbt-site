import pandas as pd
import numpy as np

# Test with exact series from bug report
series = pd.Series([
    1.000000e+00,
    1.605551e-178,
    -2.798597e-225,
    -2.225074e-308,
    -2.798597e-225,
])

rolling_mean = series.rolling(window=2).mean()

print("Testing each window:")
for i in range(1, len(series)):
    window = series.iloc[i-1:i+1]
    expected = window.mean()
    actual = rolling_mean.iloc[i]

    print(f"\nWindow at index {i}:")
    print(f"  Values: {window.values}")
    print(f"  Expected mean: {expected}")
    print(f"  Actual rolling mean: {actual}")

    if not np.isnan(actual):
        window_min = window.min()
        window_max = window.max()
        is_valid = window_min <= actual <= window_max

        if expected != 0:
            rel_error = abs(actual - expected) / abs(expected)
        else:
            rel_error = abs(actual - expected) if expected == 0 else float('inf')

        print(f"  Min: {window_min}, Max: {window_max}")
        print(f"  Within bounds? {is_valid}")
        print(f"  Relative error: {rel_error:.10f} ({rel_error*100:.2f}%)")

        if not is_valid:
            print("  *** BUG: Mean is outside window bounds! ***")

# Now test what happens with smaller window size on same problematic values
print("\n" + "="*50)
print("Testing window=3 with more values around subnormal range:")
series2 = pd.Series([
    1.0,
    1e-200,
    -2.798597e-225,
    -2.225074e-308,
    -2.798597e-225,
    1e-300,
])

for window_size in [2, 3, 4]:
    print(f"\nWindow size = {window_size}")
    rolling = series2.rolling(window=window_size).mean()

    for i in range(window_size-1, len(series2)):
        window = series2.iloc[i-window_size+1:i+1]
        expected = window.mean()
        actual = rolling.iloc[i]

        if not np.isnan(actual):
            is_valid = window.min() <= actual <= window.max()
            if not is_valid:
                print(f"  Index {i}: INVALID - mean {actual} not in [{window.min()}, {window.max()}]")
                print(f"    Window: {window.values}")
                print(f"    Expected: {expected}")