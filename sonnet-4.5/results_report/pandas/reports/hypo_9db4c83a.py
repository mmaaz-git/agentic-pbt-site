from hypothesis import given, strategies as st, settings
from pandas.tseries.api import guess_datetime_format
from datetime import datetime

DATETIME_FORMATS = [
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
]

@given(
    st.datetimes(min_value=datetime(1900, 1, 1), max_value=datetime(2100, 12, 31)),
    st.sampled_from(DATETIME_FORMATS),
    st.booleans()
)
@settings(max_examples=500)
def test_guess_datetime_format_roundtrip(dt, fmt, dayfirst):
    dt_str = dt.strftime(fmt)
    guessed_fmt = guess_datetime_format(dt_str, dayfirst=dayfirst)

    if guessed_fmt is not None:
        parsed = datetime.strptime(dt_str, guessed_fmt)
        assert parsed.date() == dt.date(), \
            f"Round-trip failed: datetime.strptime({dt_str!r}, {guessed_fmt!r}).date() = {parsed.date()} != {dt.date()}"

if __name__ == "__main__":
    test_guess_datetime_format_roundtrip()