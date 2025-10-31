import datetime
from hypothesis import given, strategies as st, example
from pydantic.deprecated.json import timedelta_isoformat
from isodate import parse_duration

@given(st.timedeltas(min_value=datetime.timedelta(days=-365),
                     max_value=datetime.timedelta(days=365)))
@example(datetime.timedelta(seconds=-1))
@example(datetime.timedelta(seconds=-30))
@example(datetime.timedelta(hours=-1))
def test_timedelta_isoformat_roundtrip(td):
    iso_string = timedelta_isoformat(td)
    parsed_td = parse_duration(iso_string)
    assert parsed_td == td, \
        f"Round-trip failed: {td} -> {iso_string} -> {parsed_td}"

if __name__ == "__main__":
    test_timedelta_isoformat_roundtrip()