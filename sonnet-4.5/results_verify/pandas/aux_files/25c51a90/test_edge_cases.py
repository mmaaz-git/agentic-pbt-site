import pandas as pd
import numpy as np
from pandas.core.methods.describe import format_percentiles, _refine_percentiles

# Test various close percentiles
test_cases = [
    ([0.01, 0.010000000000000002], "Very close floats (differ by 2e-18)"),
    ([0.01, 0.0100001], "Close floats (differ by 1e-7)"),
    ([0.01, 0.01001], "Less close floats (differ by 1e-5)"),
    ([0.01, 0.011], "Reasonably different (differ by 0.001)"),
    ([0.5, 0.50000000000000001], "Close at 50% (differ by machine epsilon)"),
    ([0.99, 0.990000000000000002], "Close at 99% (differ by machine epsilon)"),
]

print("Testing format_percentiles behavior with close values:")
print("=" * 60)

for percentiles, description in test_cases:
    print(f"\nTest: {description}")
    print(f"Inputs: {percentiles}")
    print(f"Are inputs different? {percentiles[0] != percentiles[1]}")
    print(f"Difference: {abs(percentiles[1] - percentiles[0]):.2e}")

    # Check if _refine_percentiles validates them
    try:
        refined = _refine_percentiles(percentiles)
        print(f"Refined percentiles: {refined}")
    except Exception as e:
        print(f"_refine_percentiles error: {e}")
        continue

    # Format them
    formatted = format_percentiles(percentiles)
    print(f"Formatted: {formatted}")
    print(f"Duplicates? {len(formatted) != len(set(formatted))}")

    # Try describe
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
    try:
        result = df.describe(percentiles=percentiles)
        print("describe() succeeded")
    except ValueError as e:
        print(f"describe() failed: {e}")

print("\n" + "=" * 60)
print("Testing get_precision function:")
from pandas.io.formats.format import get_precision

for percentiles, description in test_cases[:3]:
    print(f"\nPercentiles: {percentiles}")
    pcts_array = np.array(percentiles) * 100
    unique_pcts = np.unique(pcts_array)
    print(f"Unique percentiles (x100): {unique_pcts}")
    prec = get_precision(unique_pcts)
    print(f"Precision calculated: {prec}")
    print(f"Rounded with precision {prec}: {unique_pcts.round(prec)}")