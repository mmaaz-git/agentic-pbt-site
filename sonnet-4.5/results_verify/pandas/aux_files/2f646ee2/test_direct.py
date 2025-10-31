from pandas.io.sas.sas7bdat import _parse_datetime
from datetime import datetime

days = 2936549.0

print(f"Testing with days={days}")
print(f"This is {days} days from 1960-01-01")
print(f"That's approximately {days/365.25:.1f} years from 1960")
print(f"Final year would be approximately: {1960 + days/365.25:.0f}")
print(f"Python's datetime.max.year is {datetime.max.year}")
print()

try:
    result = _parse_datetime(days, 'd')
    print(f"Result: {result}")
except OverflowError as e:
    print(f"OverflowError: {e}")
except Exception as e:
    print(f"Other error ({type(e).__name__}): {e}")

# Test with edge case values
print("\nTesting edge cases:")
# Python datetime max is year 9999, 12, 31
# Days from 1960-01-01 to 9999-12-31
import datetime as dt
max_date = dt.datetime(9999, 12, 31)
base_date = dt.datetime(1960, 1, 1)
max_days_allowed = (max_date - base_date).days

print(f"Maximum days that should work: {max_days_allowed}")
print(f"Days in test: {days}")
print(f"Exceeds maximum by: {days - max_days_allowed} days")

# Test boundary
try:
    result = _parse_datetime(float(max_days_allowed), 'd')
    print(f"Max days ({max_days_allowed}) works: {result}")
except Exception as e:
    print(f"Max days failed: {e}")

try:
    result = _parse_datetime(float(max_days_allowed + 1), 'd')
    print(f"Max days + 1 ({max_days_allowed + 1}) works: {result}")
except OverflowError as e:
    print(f"Max days + 1 causes OverflowError: {e}")