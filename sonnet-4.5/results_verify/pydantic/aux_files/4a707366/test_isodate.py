import datetime
from isodate import parse_duration

# Test what the isodate library expects
test_cases = [
    "-PT1S",  # Expected for -1 second
    "-P1DT23H59M59.000000S",  # What pydantic produces
    "P-1DT23H59M59.000000S",  # Alternative with component-level negative
]

for iso_str in test_cases:
    try:
        parsed = parse_duration(iso_str)
        print(f"{iso_str:30s} -> {parsed} (total_seconds={parsed.total_seconds()})")
    except Exception as e:
        print(f"{iso_str:30s} -> ERROR: {e}")

# Test Python's internal representation
td = datetime.timedelta(seconds=-1)
print(f"\nPython timedelta(seconds=-1):")
print(f"  days={td.days}, seconds={td.seconds}, microseconds={td.microseconds}")
print(f"  total_seconds={td.total_seconds()}")