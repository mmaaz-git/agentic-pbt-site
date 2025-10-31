import pandas as pd
import numpy as np
from io import StringIO
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.pandas import data_frames, column, range_indexes

@given(
    data_frames(
        columns=[
            column("x", dtype=float),
            column("y", dtype=float),
        ],
        index=range_indexes(min_size=1, max_size=20),
    ),
    st.sampled_from(["split", "records", "index", "columns"]),
)
@settings(max_examples=200)
def test_dataframe_float_round_trip(df, orient):
    assume(not df.isnull().any().any())
    assume(not np.isinf(df.values).any())
    assume(not np.isnan(df.values).any())

    json_str = df.to_json(orient=orient)
    df_recovered = pd.read_json(StringIO(json_str), orient=orient)

    pd.testing.assert_frame_equal(df, df_recovered)

if __name__ == "__main__":
    test_dataframe_float_round_trip()