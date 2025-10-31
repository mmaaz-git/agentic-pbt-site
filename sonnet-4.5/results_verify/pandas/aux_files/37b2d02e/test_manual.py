import warnings
from pandas.io.formats.format import format_percentiles
from pandas.util._validators import validate_percentile

# Test with 1e-320
percentile = 1e-320

print(f"Testing percentile: {percentile}")

# Verify it passes validation
try:
    validate_percentile(percentile)
    print("✓ Passes validate_percentile()")
except Exception as e:
    print(f"✗ Failed validate_percentile(): {e}")

# Test format_percentiles with warnings capture
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", RuntimeWarning)
    result = format_percentiles([percentile])

    print(f"Input: {percentile}")
    print(f"Output: {result}")
    print(f"Warnings: {[str(x.message) for x in w]}")

print("\n" + "="*60 + "\n")

# Test with other subnormal floats
test_values = [2.225073858507203e-309, 5e-324, 1e-320]

for val in test_values:
    print(f"Testing: {val}")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)
        result = format_percentiles([val])
        print(f"  Result: {result}")
        if w:
            print(f"  Warnings: {len(w)} warnings")