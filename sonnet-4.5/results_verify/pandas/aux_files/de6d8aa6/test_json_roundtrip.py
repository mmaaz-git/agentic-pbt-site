from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes
import pandas as pd
import tempfile
import os


@given(
    data_frames(
        columns=[
            column("int_col", dtype=int),
            column("float_col", dtype=float),
            column("str_col", dtype=str),
        ],
        index=range_indexes(min_size=0, max_size=100),
    )
)
@settings(max_examples=100)
def test_json_round_trip_orient_split(df):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        temp_path = f.name

    try:
        df.to_json(temp_path, orient="split")
        result = pd.read_json(temp_path, orient="split")
        pd.testing.assert_frame_equal(df, result)
        print(f"Test passed for DataFrame with {len(df)} rows")
    except Exception as e:
        print(f"Test failed for DataFrame with {len(df)} rows")
        print(f"Original index type: {type(df.index)}")
        print(f"Original index: {df.index}")
        print(f"Result index type: {type(result.index)}")
        print(f"Result index: {result.index}")
        print(f"Error: {e}")
        raise
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    test_json_round_trip_orient_split()