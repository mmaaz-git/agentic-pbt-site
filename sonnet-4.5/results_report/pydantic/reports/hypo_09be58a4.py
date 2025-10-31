import datetime
import re
from hypothesis import given, strategies as st
from pydantic.deprecated.json import timedelta_isoformat


def parse_iso_duration(iso_string):
    match = re.match(r'^(-?)P(\d+)DT(\d+)H(\d+)M(\d+)\.(\d+)S$', iso_string)
    if not match:
        raise ValueError(f"Invalid ISO duration: {iso_string}")

    negative = match.group(1) == '-'
    days = int(match.group(2))
    hours = int(match.group(3))
    minutes = int(match.group(4))
    seconds = int(match.group(5))
    microseconds = int(match.group(6))

    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds + microseconds / 1_000_000

    if negative:
        total_seconds = -total_seconds

    return total_seconds


@given(st.timedeltas(
    min_value=datetime.timedelta(days=-999999),
    max_value=datetime.timedelta(days=999999)
))
def test_timedelta_isoformat_roundtrip_value(td):
    iso_str = timedelta_isoformat(td)

    reconstructed_seconds = parse_iso_duration(iso_str)
    original_seconds = td.total_seconds()

    assert abs(reconstructed_seconds - original_seconds) < 1e-6, (
        f"ISO format doesn't preserve value!\n"
        f"  Original: {td} ({original_seconds} seconds)\n"
        f"  ISO: {iso_str}\n"
        f"  Reconstructed: {reconstructed_seconds} seconds\n"
        f"  Difference: {reconstructed_seconds - original_seconds} seconds"
    )


if __name__ == "__main__":
    test_timedelta_isoformat_roundtrip_value()