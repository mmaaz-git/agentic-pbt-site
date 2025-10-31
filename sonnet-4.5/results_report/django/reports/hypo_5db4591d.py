import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.utils.dateparse import parse_time


def split_tzname_delta(tzname):
    """
    Split a time zone name into a 3-tuple of (name, sign, offset).
    This is a copy of the Django function for testing.
    """
    for sign in ["+", "-"]:
        if sign in tzname:
            name, offset = tzname.rsplit(sign, 1)
            if offset and parse_time(offset):
                if ":" not in offset:
                    offset = f"{offset}:00"
                return name, sign, offset
    return tzname, None, None


@given(
    st.text(min_size=1, max_size=30, alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_/'),
    st.sampled_from(['+', '-']),
    st.integers(min_value=0, max_value=23),
    st.integers(min_value=0, max_value=59)
)
@settings(max_examples=100)
def test_split_tzname_delta_hhmm_format_produces_invalid_offset(tzname, sign, hours, minutes):
    offset_hhmm = f"{hours:02d}{minutes:02d}"
    input_tz = f"{tzname}{sign}{offset_hhmm}"
    name, parsed_sign, parsed_offset = split_tzname_delta(input_tz)

    if parsed_offset is not None:
        expected_offset = f"{hours:02d}:{minutes:02d}"
        assert parsed_offset == expected_offset, (
            f"split_tzname_delta should format offset as HH:MM, not HHMM:00. "
            f"Input: {input_tz!r}, Expected offset: {expected_offset!r}, Got: {parsed_offset!r}"
        )


# Run the test
if __name__ == "__main__":
    print("Running Hypothesis property-based test...")
    print("Testing split_tzname_delta function with HHMM format inputs")
    print("=" * 60)

    try:
        test_split_tzname_delta_hhmm_format_produces_invalid_offset()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nThis demonstrates the bug where HHMM format produces HHMM:00 instead of HH:MM")