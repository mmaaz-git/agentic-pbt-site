from hypothesis import given, settings
from hypothesis.extra.pandas import data_frames, column
import pandas as pd
import pandas.api.interchange as interchange


@given(data_frames([
    column('A', dtype=int),
    column('B', dtype=float),
    column('C', dtype=str),
]))
@settings(max_examples=200)
def test_from_dataframe_round_trip(df):
    """
    Property: from_dataframe(df.__dataframe__()) should equal df for pandas DataFrames
    Evidence: The docstring shows this is the intended usage pattern
    """
    interchange_obj = df.__dataframe__()
    result = interchange.from_dataframe(interchange_obj)
    pd.testing.assert_frame_equal(result, df)

# Run the test
if __name__ == "__main__":
    test_from_dataframe_round_trip()