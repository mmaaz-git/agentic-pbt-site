import datetime
from pydantic.deprecated.json import timedelta_isoformat
from isodate import parse_duration

# Test with -1 second
td_original = datetime.timedelta(seconds=-1)
print(f"Original: {td_original} (total_seconds={td_original.total_seconds()})")

iso_string = timedelta_isoformat(td_original)
print(f"ISO format: {iso_string}")

td_parsed = parse_duration(iso_string)
print(f"Parsed: {td_parsed} (total_seconds={td_parsed.total_seconds()})")

print(f"Are they equal? {td_original == td_parsed}")