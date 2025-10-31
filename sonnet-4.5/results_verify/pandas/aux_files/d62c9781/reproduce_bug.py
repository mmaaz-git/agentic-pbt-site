import pandas as pd
from pandas.io.sas.sas7bdat import _parse_datetime, _convert_datetimes

value = 0.5

parse_result = _parse_datetime(value, "d")
print(f"_parse_datetime(0.5, 'd') = {parse_result}")

convert_result = _convert_datetimes(pd.Series([value]), "d").iloc[0]
print(f"_convert_datetimes([0.5], 'd') = {convert_result}")

diff = pd.Timestamp(parse_result) - pd.Timestamp(convert_result)
print(f"Difference: {diff}")

# Also test with other fractional values
print("\n=== Testing with other fractional values ===")
test_values = [0.25, 0.5, 0.75, 1.5, 2.33]

for val in test_values:
    parse_res = _parse_datetime(val, "d")
    convert_res = _convert_datetimes(pd.Series([val]), "d").iloc[0]
    diff = pd.Timestamp(parse_res) - pd.Timestamp(convert_res)
    print(f"Value {val}: parse={parse_res}, convert={convert_res}, diff={diff}")