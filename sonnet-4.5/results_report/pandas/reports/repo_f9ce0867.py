import pandas as pd
from pandas.io.sas.sas7bdat import _parse_datetime, _convert_datetimes

# Test cases with fractional days
test_values = [0.25, 0.5, 0.75, 1.5, 2.33]

print("Demonstrating the bug: _convert_datetimes truncates fractional days\n")
print("="*70)

for value in test_values:
    # Test _parse_datetime (scalar function)
    parse_result = _parse_datetime(value, "d")

    # Test _convert_datetimes (vectorized function)
    series = pd.Series([value])
    convert_result = _convert_datetimes(series, "d").iloc[0]

    # Calculate the difference
    parse_ts = pd.Timestamp(parse_result)
    convert_ts = pd.Timestamp(convert_result)
    diff = parse_ts - convert_ts

    print(f"Input value: {value} days")
    print(f"  _parse_datetime result:    {parse_result}")
    print(f"  _convert_datetimes result: {convert_result}")
    print(f"  Difference (data lost):    {diff}")
    print(f"  Hours lost: {diff.total_seconds() / 3600:.2f}")
    print("-"*70)

print("\nSummary: All fractional day components are silently truncated!")
print("This causes significant data loss when reading SAS files with fractional days.")