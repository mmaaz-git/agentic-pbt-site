import io
import pandas as pd
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes


@settings(max_examples=500)
@given(
    df=data_frames(
        columns=[
            column("a", dtype=int),
            column("b", dtype=float),
        ],
        index=range_indexes(min_size=0, max_size=20),
    )
)
def test_json_roundtrip_split_preserves_index_dtype(df):
    json_str = df.to_json(orient="split")
    result = pd.read_json(io.StringIO(json_str), orient="split")
    assert df.index.dtype == result.index.dtype, f"Original dtype: {df.index.dtype}, Result dtype: {result.index.dtype}"

if __name__ == "__main__":
    test_json_roundtrip_split_preserves_index_dtype()