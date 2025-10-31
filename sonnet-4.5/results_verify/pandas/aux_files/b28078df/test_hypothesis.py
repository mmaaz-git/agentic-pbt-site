import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes

@given(
    data_frames(
        columns=[
            column("group", elements=st.integers(min_value=0, max_value=5)),
            column("value", elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)),
        ],
        index=range_indexes(min_size=1, max_size=100)
    )
)
@settings(max_examples=200)
def test_mean_between_min_and_max(df):
    grouped = df.groupby("group")
    min_result = grouped["value"].min()
    max_result = grouped["value"].max()
    mean_result = grouped["value"].mean()

    for group_name in mean_result.index:
        min_val = min_result[group_name]
        max_val = max_result[group_name]
        mean_val = mean_result[group_name]

        assert min_val <= mean_val <= max_val, \
            f"mean {mean_val} not between min {min_val} and max {max_val} for group {group_name}"

if __name__ == "__main__":
    test_mean_between_min_and_max()
    print("Hypothesis test completed")