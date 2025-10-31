from hypothesis import given, strategies as st, settings, assume
from io import StringIO
import pandas as pd

@settings(max_examples=200)
@given(
    st.data(),
    st.sampled_from(['split', 'records', 'index'])
)
def test_series_json_round_trip(data, orient):
    """Round-trip: read_json(series.to_json(orient=x), orient=x, typ='series') should preserve data"""

    nrows = data.draw(st.integers(min_value=1, max_value=10))
    values = data.draw(st.lists(
        st.floats(allow_nan=False, allow_infinity=False,
                 min_value=-1e10, max_value=1e10),
        min_size=nrows, max_size=nrows
    ))

    series = pd.Series(values)

    if orient == 'index':
        assume(series.index.is_unique)

    json_str = series.to_json(orient=orient)
    recovered = pd.read_json(StringIO(json_str), orient=orient, typ='series')

    pd.testing.assert_series_equal(recovered, series, check_index_type=False)

if __name__ == "__main__":
    test_series_json_round_trip()