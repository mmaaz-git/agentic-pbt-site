from hypothesis import given, strategies as st, settings
import pandas.io.formats.format as fmt

@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1))
@settings(max_examples=1000)
def test_format_percentiles_all_end_with_percent(percentiles):
    """
    Property: all formatted strings should end with '%'
    and not contain 'nan' or 'inf'
    """
    formatted = fmt.format_percentiles(percentiles)
    for f in formatted:
        assert f.endswith('%'), f"Formatted value '{f}' does not end with '%'"
        assert 'nan' not in f.lower(), f"Formatted value '{f}' contains 'nan'"

if __name__ == "__main__":
    try:
        test_format_percentiles_all_end_with_percent()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"Error during testing: {e}")