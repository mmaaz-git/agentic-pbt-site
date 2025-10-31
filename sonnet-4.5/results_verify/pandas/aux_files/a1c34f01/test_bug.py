from hypothesis import given, strategies as st, settings
import pandas.io.formats.format as fmt
import warnings
import pytest


@given(
    value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    count=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=500)
def test_format_percentiles_no_warnings_for_duplicates(value, count):
    """When all percentiles are the same, the function should not produce warnings."""
    percentiles = [value] * count

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = fmt.format_percentiles(percentiles)

        runtime_warnings = [warning for warning in w
                          if issubclass(warning.category, RuntimeWarning)]

        assert len(runtime_warnings) == 0, \
            f"Should not produce warnings for duplicate values: {[str(w.message) for w in runtime_warnings]}"


# Run the test
if __name__ == "__main__":
    # Test with specific failing input mentioned in the report
    test_format_percentiles_no_warnings_for_duplicates(0.0, 2)
    print("Test with value=0.0, count=2 passed (no assertion error)")