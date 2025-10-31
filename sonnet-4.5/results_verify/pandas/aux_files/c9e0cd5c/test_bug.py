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
    split_result = df.to_dict(orient='split')
    tight_result = df.to_dict(orient='tight')

    assert split_result['data'] == tight_result['data']

# Run the test
if __name__ == '__main__':
    test_to_dict_tight_should_use_computed_data()
    print("Test completed successfully!")