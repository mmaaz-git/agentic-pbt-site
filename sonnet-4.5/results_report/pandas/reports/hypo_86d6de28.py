from hypothesis import given, settings
from hypothesis.extra.pandas import data_frames, column
from hypothesis import strategies as st
import pandas as pd


@settings(max_examples=200)
@given(data_frames([
    column('int_col', dtype=int),
    column('float_col', dtype=float),
    column('str_col', dtype=str)
], index=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=20, unique=True)))
def test_to_dict_tight_should_use_computed_data(df):
    """Test that 'split' and 'tight' orientations produce identical data values."""
    split_result = df.to_dict(orient='split')
    tight_result = df.to_dict(orient='tight')

    assert split_result['data'] == tight_result['data'], \
        f"Data mismatch: split and tight should produce identical data values"

    # Also verify the structural differences are as expected
    assert 'index_names' in tight_result, "'tight' should include 'index_names'"
    assert 'column_names' in tight_result, "'tight' should include 'column_names'"
    assert 'index_names' not in split_result, "'split' should not include 'index_names'"
    assert 'column_names' not in split_result, "'split' should not include 'column_names'"

if __name__ == "__main__":
    # Run the test
    test_to_dict_tight_should_use_computed_data()
    print("All property-based tests passed successfully!")