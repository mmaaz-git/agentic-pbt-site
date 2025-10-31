from pandas.tseries.api import guess_datetime_format
from datetime import datetime

dt_str = "2000-01-02"
guessed_fmt = guess_datetime_format(dt_str, dayfirst=True)

print(f"Input: {dt_str}")
print(f"Guessed format: {guessed_fmt}")

parsed = datetime.strptime(dt_str, guessed_fmt)
print(f"Parsed date: {parsed.date()}")
print(f"Expected date: 2000-01-02")

assert parsed.date() == datetime(2000, 1, 2).date(), f"Got {parsed.date()} instead of 2000-01-02"