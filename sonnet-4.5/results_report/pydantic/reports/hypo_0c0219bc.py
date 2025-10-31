import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

import datetime
from hypothesis import given, strategies as st
from pydantic.deprecated.json import timedelta_isoformat


def parse_iso_duration_to_seconds(iso_string):
    is_negative = iso_string.startswith('-')
    if is_negative:
        iso_string = iso_string[1:]

    if not iso_string.startswith('P'):
        raise ValueError("Invalid ISO duration")

    iso_string = iso_string[1:]
    days = hours = minutes = 0
    seconds = 0.0

    if 'D' in iso_string:
        day_part, iso_string = iso_string.split('D', 1)
        days = int(day_part)

    if 'T' in iso_string:
        iso_string = iso_string[1:]
        if 'H' in iso_string:
            hour_part, iso_string = iso_string.split('H', 1)
            hours = int(hour_part)
        if 'M' in iso_string:
            minute_part, iso_string = iso_string.split('M', 1)
            minutes = int(minute_part)
        if 'S' in iso_string:
            second_part = iso_string.split('S')[0]
            seconds = float(second_part)

    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if is_negative:
        total_seconds = -total_seconds

    return total_seconds


@given(st.integers(min_value=-1000000, max_value=1000000))
def test_timedelta_isoformat_semantic_correctness(seconds):
    """Property: ISO format output should represent the same duration as the input."""
    td = datetime.timedelta(seconds=seconds)
    iso_output = timedelta_isoformat(td)

    expected_seconds = td.total_seconds()
    parsed_seconds = parse_iso_duration_to_seconds(iso_output)

    assert abs(expected_seconds - parsed_seconds) < 0.001, \
        f"ISO format {iso_output} represents {parsed_seconds}s but should represent {expected_seconds}s"


if __name__ == "__main__":
    test_timedelta_isoformat_semantic_correctness()