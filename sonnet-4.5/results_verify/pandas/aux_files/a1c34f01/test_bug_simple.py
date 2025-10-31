import pandas.io.formats.format as fmt
import warnings

# Test with specific failing input mentioned in the report
def test_duplicate_values():
    percentiles = [0.0, 0.0]  # Duplicate values

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = fmt.format_percentiles(percentiles)

        runtime_warnings = [warning for warning in w
                          if issubclass(warning.category, RuntimeWarning)]

        print(f"Result: {result}")
        print(f"Number of RuntimeWarnings: {len(runtime_warnings)}")
        for warning in runtime_warnings:
            print(f"  - {warning.message}")

test_duplicate_values()

print("\nTesting with [0.5, 0.5, 0.5]:")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = fmt.format_percentiles([0.5, 0.5, 0.5])

    runtime_warnings = [warning for warning in w
                      if issubclass(warning.category, RuntimeWarning)]

    print(f"Result: {result}")
    print(f"Number of RuntimeWarnings: {len(runtime_warnings)}")
    for warning in runtime_warnings:
        print(f"  - {warning.message}")