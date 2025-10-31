from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames
import pandas as pd

@given(data_frames([
    column('a', dtype=int),
    column('b', dtype=float),
    column('c', dtype=str),
]))
@settings(max_examples=100)
def test_dataframe_to_dict_from_dict_roundtrip_dict_orient(df):
    result = pd.DataFrame.from_dict(df.to_dict(orient='dict'))
    pd.testing.assert_frame_equal(result, df)

if __name__ == "__main__":
    test_dataframe_to_dict_from_dict_roundtrip_dict_orient()
    print("Test completed!")