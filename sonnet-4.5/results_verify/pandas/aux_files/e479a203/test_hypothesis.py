from hypothesis import given, settings, example
from hypothesis.extra.pandas import column, data_frames, range_indexes
import pandas as pd
import numpy as np
import io

@given(
    df=data_frames([
        column('a', dtype=int),
        column('b', dtype=float),
    ], index=range_indexes(min_size=0, max_size=50))
)
@example(pd.DataFrame({'a': [0], 'b': [np.finfo(np.float64).max]}))
@settings(max_examples=100)
def test_to_json_read_json_roundtrip(df):
    json_buffer = io.StringIO()
    df.to_json(json_buffer, orient='records')
    json_buffer.seek(0)
    result = pd.read_json(json_buffer, orient='records')

    if len(df) > 0:
        pd.testing.assert_frame_equal(
            df.reset_index(drop=True),
            result.reset_index(drop=True),
            check_dtype=False
        )

if __name__ == '__main__':
    test_to_json_read_json_roundtrip()
    print("Test completed without errors")