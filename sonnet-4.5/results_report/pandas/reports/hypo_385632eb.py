from hypothesis import given, strategies as st, settings, example
from pandas.core.methods.describe import format_percentiles

@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=50))
@example([0.0, 3.6340605919844266e-284])  # Add the specific failing case
@settings(max_examples=100)
def test_format_percentiles_different_inputs_remain_different(percentiles):
    """
    Property: If two percentiles differ, they should have different formatted strings
    """
    unique_percentiles = list(set(percentiles))

    if len(unique_percentiles) <= 1:
        return

    formatted = format_percentiles(percentiles)
    unique_formatted = set(formatted)

    if len(unique_percentiles) > 1:
        assert len(unique_formatted) > 1, (
            f"Different percentiles collapsed to same format: "
            f"input had {len(unique_percentiles)} unique values, "
            f"but output has only {len(unique_formatted)} unique strings: {unique_formatted}"
        )

if __name__ == "__main__":
    # Run the test
    test_format_percentiles_different_inputs_remain_different()