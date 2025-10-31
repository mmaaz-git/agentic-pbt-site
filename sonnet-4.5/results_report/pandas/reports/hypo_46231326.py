from io import StringIO
import pandas as pd
from hypothesis import given, settings, strategies as st


@given(st.lists(st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text()), min_size=1, max_size=20))
@settings(max_examples=500)
def test_series_roundtrip_split(data):
    s = pd.Series(data)
    json_str = s.to_json(orient='split')
    s_recovered = pd.read_json(StringIO(json_str), typ='series', orient='split')

    pd.testing.assert_series_equal(s, s_recovered, check_dtype=False)


if __name__ == "__main__":
    test_series_roundtrip_split()