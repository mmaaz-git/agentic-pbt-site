from pandas.tseries.api import guess_datetime_format
from datetime import datetime

test_cases = [
    "2000-01-02",
    "01/02/2000",
    "02/01/2000",
    "2000/01/02",
]

for dt_str in test_cases:
    print(f"\nInput: {dt_str}")

    # Test with dayfirst=False
    fmt_false = guess_datetime_format(dt_str, dayfirst=False)
    print(f"  dayfirst=False: {fmt_false}")
    if fmt_false:
        try:
            parsed = datetime.strptime(dt_str, fmt_false)
            print(f"    Parsed: {parsed.date()}")
        except:
            print(f"    Failed to parse")

    # Test with dayfirst=True
    fmt_true = guess_datetime_format(dt_str, dayfirst=True)
    print(f"  dayfirst=True: {fmt_true}")
    if fmt_true:
        try:
            parsed = datetime.strptime(dt_str, fmt_true)
            print(f"    Parsed: {parsed.date()}")
        except:
            print(f"    Failed to parse")