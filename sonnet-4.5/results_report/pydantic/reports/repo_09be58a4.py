import datetime
from pydantic.deprecated.json import timedelta_isoformat

td = datetime.timedelta(days=-1, microseconds=1)

print(f"Timedelta: {td}")
print(f"Total seconds: {td.total_seconds()}")
print(f"Days component: {td.days}")
print(f"Seconds component: {td.seconds}")
print(f"Microseconds component: {td.microseconds}")

iso_str = timedelta_isoformat(td)
print(f"\nISO format: {iso_str}")

print(f"\nExpected total seconds: {td.total_seconds()}")
print(f"ISO string represents: -(1 day + 0 hours + 0 minutes + 0.000001 seconds)")
print(f"Which equals: -86400.000001 seconds")
print(f"Difference from expected: {-86400.000001 - td.total_seconds()} seconds")