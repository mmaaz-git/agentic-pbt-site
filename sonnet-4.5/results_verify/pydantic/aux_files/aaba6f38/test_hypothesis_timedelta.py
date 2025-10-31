import warnings
from datetime import timedelta
from hypothesis import given, strategies as st, settings


def parse_iso_duration_to_seconds(iso_str):
    if iso_str.startswith('-P'):
        is_negative = True
        iso_str = iso_str[2:]
    elif iso_str.startswith('P'):
        is_negative = False
        iso_str = iso_str[1:]
    else:
        raise ValueError(f"Invalid ISO duration: {iso_str}")

    if 'T' in iso_str:
        date_part, time_part = iso_str.split('T')
    else:
        date_part = iso_str
        time_part = ''

    days = 0
    if 'D' in date_part:
        days = int(date_part.split('D')[0])

    hours = minutes = seconds = microseconds = 0
    if time_part:
        if 'H' in time_part:
            hours = int(time_part.split('H')[0])
            time_part = time_part.split('H')[1]
        if 'M' in time_part:
            if 'S' in time_part:
                minutes_str = time_part.split('M')[0]
                if minutes_str:
                    minutes = int(minutes_str)
                time_part = time_part.split('M')[1]
            else:
                minutes = int(time_part.split('M')[0])
                time_part = ''
        if 'S' in time_part:
            sec_str = time_part.split('S')[0]
            if '.' in sec_str:
                sec_part, microsec_part = sec_str.split('.')
                seconds = int(sec_part)
                microseconds = int(microsec_part)
            else:
                seconds = int(sec_str)

    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds + microseconds / 1000000
    if is_negative:
        total_seconds = -total_seconds

    return total_seconds


@settings(max_examples=1000)
@given(st.timedeltas(min_value=timedelta(days=-365), max_value=timedelta(days=365)))
def test_timedelta_isoformat_roundtrip(td):
    from pydantic.deprecated.json import timedelta_isoformat

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iso = timedelta_isoformat(td)

    reconstructed_seconds = parse_iso_duration_to_seconds(iso)
    original_seconds = td.total_seconds()

    assert abs(reconstructed_seconds - original_seconds) < 0.000001