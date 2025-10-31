from hypothesis import given, strategies as st, settings
from datetime import datetime
from pandas.tseries.holiday import next_monday


@given(st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2030, 1, 1)))
@settings(max_examples=200)
def test_next_monday_name_contract(dt):
    result = next_monday(dt)

    if dt.weekday() in [1, 2, 3, 4]:
        assert result.weekday() == 0, \
            f"next_monday from {dt.strftime('%A')} should return next Monday, not {result.strftime('%A')}"

# Run the property test
print("Running property-based test...")
try:
    test_next_monday_name_contract()
    print("Test passed")
except AssertionError as e:
    print(f"Test failed: {e}")

# Test the specific failing input
print("\nTesting specific failing input:")
specific_date = datetime(2000, 2, 1, 0, 0)  # Tuesday
result = next_monday(specific_date)
print(f"Input: {specific_date.strftime('%A, %Y-%m-%d')}")
print(f"Result: {result.strftime('%A, %Y-%m-%d')}")
print(f"Input weekday: {specific_date.weekday()} (0=Monday, 6=Sunday)")
print(f"Result weekday: {result.weekday()}")

# Reproduce the example from the bug report
print("\n" + "="*50)
print("Reproducing bug report example:")
thursday = datetime(2020, 6, 4)
result = next_monday(thursday)

print(f"Input: {thursday.strftime('%A')}")
print(f"Result: {result.strftime('%A')}")
print(f"Result weekday: {result.weekday()}")

try:
    assert result.weekday() == 0
    print("Assertion passed: Result is Monday")
except AssertionError:
    print("AssertionError: Result is not Monday")

# Test all days of the week
print("\n" + "="*50)
print("Testing all days of the week:")
test_dates = [
    datetime(2020, 6, 1),  # Monday
    datetime(2020, 6, 2),  # Tuesday
    datetime(2020, 6, 3),  # Wednesday
    datetime(2020, 6, 4),  # Thursday
    datetime(2020, 6, 5),  # Friday
    datetime(2020, 6, 6),  # Saturday
    datetime(2020, 6, 7),  # Sunday
]

for date in test_dates:
    result = next_monday(date)
    print(f"{date.strftime('%A'):10} -> {result.strftime('%A'):10} (weekday {result.weekday()})")