import datetime
import warnings
import re
from hypothesis import given, strategies as st, settings
from pydantic.deprecated.json import timedelta_isoformat

def parse_iso_duration(iso_str):
    is_negative = iso_str.startswith('-')
    if is_negative:
        iso_str = iso_str[1:]

    match = re.match(r'P(\d+)DT(\d+)H(\d+)M(\d+)\.(\d+)S', iso_str)
    if not match:
        return None

    days, hours, minutes, seconds, microseconds = map(int, match.groups())
    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds + microseconds / 1000000

    if is_negative:
        total_seconds = -total_seconds

    return total_seconds

@given(st.timedeltas())
@settings(max_examples=1000)
def test_timedelta_isoformat_value_preservation(td):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = timedelta_isoformat(td)

    parsed_total_seconds = parse_iso_duration(result)
    assert parsed_total_seconds is not None, f"Failed to parse: {result}"

    actual_total_seconds = td.total_seconds()

    assert abs(parsed_total_seconds - actual_total_seconds) < 1e-6, (
        f"Value mismatch for {td}:\n"
        f"  ISO format: {result}\n"
        f"  Parsed total seconds: {parsed_total_seconds}\n"
        f"  Actual total seconds: {actual_total_seconds}\n"
        f"  Difference: {abs(parsed_total_seconds - actual_total_seconds)}"
    )

if __name__ == "__main__":
    test_timedelta_isoformat_value_preservation()