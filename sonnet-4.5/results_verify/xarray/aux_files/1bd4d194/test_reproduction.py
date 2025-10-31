import datetime
from pydantic.deprecated.json import timedelta_isoformat

td = datetime.timedelta(days=-1, microseconds=1)
result = timedelta_isoformat(td)

print(f"Input: {td}")
print(f"Total seconds: {td.total_seconds()}")
print(f"ISO output: {result}")
print(f"Expected: approximately -86399.999999 seconds")
print(f"Actual interpretation: -86400.000001 seconds")

# Let's also inspect the timedelta object components
print(f"\nTimedelta components:")
print(f"  days: {td.days}")
print(f"  seconds: {td.seconds}")
print(f"  microseconds: {td.microseconds}")

# Let's verify more examples
print("\nAdditional test cases:")
test_cases = [
    datetime.timedelta(days=-1, seconds=1),
    datetime.timedelta(days=-2, microseconds=1000000),  # -2 days + 1 second
    datetime.timedelta(days=-1, seconds=3600),  # -1 day + 1 hour
]

for td_test in test_cases:
    result = timedelta_isoformat(td_test)
    print(f"Input: {td_test}, Total seconds: {td_test.total_seconds()}, ISO: {result}")