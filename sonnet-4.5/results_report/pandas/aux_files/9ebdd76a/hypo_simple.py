from hypothesis import given, strategies as st, example
import pandas as pd
from io import StringIO
from pandas.testing import assert_frame_equal

@given(
    data=st.lists(
        st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.one_of(st.integers(), st.text()),
            min_size=1,
            max_size=5,
        ),
        min_size=1,
        max_size=20,
    ),
    orient=st.sampled_from(['records', 'columns']),
)
@example(data=[{'0': 0}], orient='records')
def test_read_json_to_json_roundtrip(data, orient):
    df = pd.DataFrame(data)
    json_str = df.to_json(orient=orient)
    df_back = pd.read_json(StringIO(json_str), orient=orient)
    assert_frame_equal(df, df_back)

if __name__ == "__main__":
    # Run with the specific failing example
    test_read_json_to_json_roundtrip()