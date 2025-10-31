from hypothesis import given, strategies as st
from pandas.plotting._matplotlib.converter import TimeFormatter


@given(st.floats(min_value=86400, max_value=172800, allow_nan=False, allow_infinity=False))
def test_timeformatter_wraps_at_24_hours(seconds_since_midnight):
    formatter = TimeFormatter(locs=[])
    result = formatter(seconds_since_midnight)

    s = int(seconds_since_midnight)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    _, h = divmod(h, 24)

    assert 0 <= h < 24

if __name__ == "__main__":
    test_timeformatter_wraps_at_24_hours()