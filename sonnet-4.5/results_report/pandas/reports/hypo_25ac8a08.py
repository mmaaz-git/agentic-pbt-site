from hypothesis import given, strategies as st
import pandas as pd
from pandas.api.interchange import from_dataframe


@given(st.lists(st.one_of(st.booleans(), st.none()), min_size=1, max_size=100))
def test_round_trip_nullable_bool(bool_list):
    df = pd.DataFrame({"col": pd.array(bool_list, dtype="boolean")})
    result = from_dataframe(df.__dataframe__())
    pd.testing.assert_frame_equal(result, df)

if __name__ == "__main__":
    test_round_trip_nullable_bool()