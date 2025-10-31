import pandas as pd
import warnings

series = pd.Series([1, 2, 3, 4, 5])

# Test with subnormal float
print("Testing Series.describe() with subnormal percentile:")
print("percentiles=[2.225073858507203e-309]")
print()

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", RuntimeWarning)
    result = series.describe(percentiles=[2.225073858507203e-309])
    print(result)

    if w:
        print(f"\nWarnings generated: {len(w)}")
        for warning in w:
            print(f"  - {warning.message}")

print("\n" + "="*60 + "\n")

# Test with 5e-324
print("Testing Series.describe() with percentile 5e-324:")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", RuntimeWarning)
    result = series.describe(percentiles=[5e-324])
    print(result)

    if w:
        print(f"\nWarnings generated: {len(w)}")
        for warning in w:
            print(f"  - {warning.message}")