from pandas.io.sas.sas7bdat import _parse_datetime
from datetime import datetime, timedelta

# Find the exact boundary
base = datetime(1960, 1, 1)
max_date = datetime(9999, 12, 31, 23, 59, 59, 999999)
max_days = (max_date - base).total_seconds() / 86400.0

print(f"Base date: {base}")
print(f"Max possible date: {max_date}")
print(f"Max days (with fractional part): {max_days}")
print()

# Test around the boundary
test_values = [
    max_days - 1.0,
    max_days - 0.5,
    max_days - 0.1,
    max_days,
    max_days + 0.000001,
    max_days + 0.1,
    max_days + 0.5,
    max_days + 1.0,
    2936550.0  # Value mentioned in hypothesis
]

for days in test_values:
    try:
        result = _parse_datetime(days, 'd')
        print(f"Days {days:15.6f} -> Success: {result}")
    except OverflowError as e:
        print(f"Days {days:15.6f} -> OverflowError: {e}")
    except Exception as e:
        print(f"Days {days:15.6f} -> {type(e).__name__}: {e}")