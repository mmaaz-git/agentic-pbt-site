from hypothesis import given, settings, strategies as st
import warnings
import pytest
from pandas.io.formats.format import format_percentiles


@given(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False), min_size=1, max_size=20, unique=True))
@settings(max_examples=1000)
def test_format_percentiles_no_warnings(percentiles):
    """
    Property: format_percentiles should not generate RuntimeWarnings for valid inputs.
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)
        result = format_percentiles(percentiles)

        runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]

        if runtime_warnings:
            warning_messages = [str(warning.message) for warning in runtime_warnings]
            pytest.fail(
                f"format_percentiles generated RuntimeWarnings for percentiles {percentiles}:\n" +
                "\n".join(warning_messages)
            )

if __name__ == "__main__":
    test_format_percentiles_no_warnings()