from hypothesis import given, strategies as st, settings
import pandas as pd
from pandas.io.json import read_json, to_json
import io

@given(
    data=st.lists(
        st.fixed_dictionaries({
            'a': st.integers(min_value=-1000, max_value=1000),
            'b': st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        }),
        min_size=0,
        max_size=20
    )
)
@settings(max_examples=100)
def test_roundtrip_orient_records(data):
    df = pd.DataFrame(data)
    json_str = to_json(None, df, orient='records')
    result = read_json(io.StringIO(json_str), orient='records')
    pd.testing.assert_frame_equal(result, df)

if __name__ == "__main__":
    test_roundtrip_orient_records()
    print("Test completed")